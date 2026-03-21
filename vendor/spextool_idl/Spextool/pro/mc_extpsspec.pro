;+
; NAME:
;     mc_extpsspec
;
; PURPOSE:
;     (Optimally) extracts point source spectra from a XD spectral image.
;
; CATEGORY:
;     Spectroscopy
;
; CALLING SEQUENCE:
;     mc_extpsspec,image,var,bitmask,omask,orders,wavecal,spatcal, $
;                  tracecoeffs,apradius,apsign,BPMASK=bpmask,BGINFO=bginfo, $
;                  OPTINFO=optinfo,BPFINFO=bpfinfo,UPDATE=update, $
;                  WIDGET_ID=widget_id,CANCEL=cancel
;
; INPUTS:
;     image       - A 2-D image 
;     var         - A 2-D variance image
;     bitmask     - 
;     norders     - The number of orders
;     naps        - The number of apertures
;     xmin        - The minimum xrange value for ALL orders, not just
;                   those being extracted.
;     xmax        - The maximum xrange value for ALL orders, not just
;                   those being extracted.
;     xranges     - An array [2,norders] of pixel positions where the
;                   orders are completely on the array
;     slith_arc   - Slit length in arcsecs.
;     apradii     - Array of aperture radii in arcseconds
;     apsign      - Array of 1s and -1s indicating which apertures
;                   are positive and which are negative (for IR pair
;                   subtraction). 
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     SPATCOEFF - A structure will norders elements each of which will 
;                 contain an array of coefficients for each "row" of
;                 the spatial map.
;     PSBGINFO  - The background info for standard point source
;                 background definitions [bgstart,bgwidth].
;                 bgstart - The radius in arcseconds at which to start the
;                           background region definition (see mkmask_ps)
;                 bgwidth - The width of the background region in arcseconds
;                           (see mkmask_ps)
;     XSBGINFO  - (Background Regions) Array of background regions 
;                 in arcseconds ([[1,2],[13-15]]).
;     PSFWIDTH  - The radius at which the profile goes to zero.
;     BGORDER   - Polynomial fit degree order of the background.  If
;                 omitted, then the background is not subtracted.
;     BGSUBIMG  - The background subtracted image 
;     UPDATE    - If set, the program will launch the Fanning
;                 showprogress widget.
;     WIDGET_ID - If given, a cancel button is added to the Fanning
;                 showprogress routine.  The widget blocks the
;                 WIDGET_ID, and checks for for a user cancel
;                 command.
;     CANCEL    - Set on return if there is a problem
;
; OUTPUTS:
;     Returns an (stop-start+1,naps,norders) array of spectra
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None   
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     Later     
;
; EXAMPLE:
;
; MODIFICATION HISTORY:
;     2000-08-24 - Written by M. Cushing, Institute for Astronomy, UH
;     2001-10-04 - Added xranges input
;     2002-11-20 - Added optimal extraction.  Heavily modified.
;     2005-05-02 - Added the BGSUBIMG keyword
;     2005-10-10 - Modified background subtraction for BG degrees of
;                  zero to avoid bad pixel clusters when using small
;                  numbers of BG pixels
;     2007-07-16 - Removed start and stop input
;     2008-08-11 - Removed BGSTART and BGWIDTH keywords and replaced
;                  with PSBGINFO.  Also added XSBGINFO keyword.
;     2009-11-02 - Added the xmin and xmax inputs.
;     2015-01-05 - Added BITMASK keyword and removed BDPXMK keyword.
;     2017-08-01 - Major rewrite as a part of Spextool 5.0.
;     2019-06-24 - Included a check to avoid edges of the array where
;                  the resampled bad pixel mask goes nearly all bad.
;
;-
function mc_extpsspec,image,var,bitmask,omask,orders,wavecal,spatcal, $
                      tracecoeffs,apradius,apsign,BPMASK=bpmask,BGINFO=bginfo, $
                      OPTINFO=optinfo,BPFINFO=bpfinfo,UPDATE=update, $
                      WIDGET_ID=widget_id,CANCEL=cancel

  cancel = 0

;  Set up debug variables
  
  debugbgsub = 0
  debugbpf = 0
  debugopt = 0
  debugprofmod = 0
  debugxrange = [1060,1061]

;  Check to see what we are doing

  doopt = keyword_set(OPTINFO)
  dobpf = keyword_set(BPFINFO)*(~doopt)  ; only do it optimal is not requested
  dobgsub = keyword_SET(BGINFO)

;  Make sure we are subtracting the background if we are doing optimal

  if doopt and ~dobgsub then begin

     message = 'Background subtraction must accompany optimal extraction.'
     mc_message,message,WIDGET_ID=widget_id
     cancel = 1
     return,-1
    
  endif

;  Get array sizes
  
  s       = size(image,/DIMEN)
  ncols   = s[0]
  nrows   = s[1]

  norders = n_elements(orders)
  naps    = n_elements(apsign[*,0])

;  Create new arrays
  
  xx = rebin(indgen(ncols),ncols,nrows)
  yy = rebin(reform(indgen(nrows),1,nrows),ncols,nrows)  

  if ~keyword_set(BPMASK) then bpmask = make_array(ncols,nrows,/BYTE,VALUE=1)
  
;  Set up debugging window.
  
  if debugbgsub or debugbpf or debugopt then begin
     
     window, /FREE
     wid = !d.window
     re = ' '
     
  endif

;  Get set up for the profile if need be

  if keyword_SET(BPFINFO) then begin

     imgstruc = bpfinfo.imgs
     profstruc = bpfinfo.profs
     thresh = bpfinfo.thresh
     medprof = bpfinfo.medprof

     if n_tags(bpfinfo) gt 4 then atmos = {awave:bpfinfo.awave, $
                                           atrans:bpfinfo.atrans}
     
  endif

  if keyword_set(OPTINFO) then begin

     imgstruc = optinfo.imgs
     profstruc = optinfo.profs
     thresh = optinfo.thresh
     medprof = optinfo.medprof
     psfradius = optinfo.psfradius

     if n_tags(optinfo) gt 5 then atmos = {awave:optinfo.awave, $
                                           atrans:optinfo.atrans}
     
  endif
  
;  Set up Fanning update object.
  
  if keyword_set(UPDATE) and keyword_set(WIDGET_ID) then begin

     cancelbutton = (n_elements(WIDGET_ID) ne 0) ? 1:0
     progressbar = obj_new('SHOWPROGRESS',widget_id,COLOR=2,$
                           CANCELBUTTON=cancelbutton,$
                           MESSAGE='Extracting spectra...')
     progressbar -> start
     
  endif
  
;  Start order loop

  m = 0
  for i = 0, norders-1 do begin

;  Generate the spatial model if requested

     if dobpf or doopt then begin
               
        spatmod = mc_mkspatmodel(imgstruc.(i),(profstruc.(i))[*,1], $
                                 tracecoeffs[*,(i*naps):(i*naps+naps-1)],$
                                 replicate(apradius,naps),ATMOS=atmos, $
                                 AVEPROF=medprof,DEBUG=debugprofmod, $
                                 CANCEL=cancel)
        if cancel then return, -1
        
     endif

;  Determine the minimum and maximum columns you are extracting over
     
     zordr = where(omask eq orders[i])
     xmin = min(xx[zordr],MAX=xmax)
     nwaves = (xmax-xmin)+1

;  Create the output arrays     

     owspec = make_array(nwaves,/DOUBLE,VALUE=!values.f_nan)
     ofspec = make_array(nwaves,naps,/DOUBLE,VALUE=!values.f_nan)
     ouspec = make_array(nwaves,naps,/DOUBLE,VALUE=!values.f_nan)
     obspec = make_array(nwaves,naps,/BYTE,VALUE=0)

;  Star the loop over wavelength

     for j = 0,nwaves-1 do begin

;  Find the slit pixels using the omask 
        
        somask = omask[xmin+j,*]
        zslit = where(somask eq orders[i],nslit)

;  Now carve out the slit in the various images
        
        slit_pix = reform(yy[xmin+j,zslit])
        slit_arc = reform(spatcal[xmin+j,zslit])
        slitimg  = reform(image[xmin+j,zslit])
        slitvar  = reform(var[xmin+j,zslit])
        slitbsm  = reform(bitmask[xmin+j,zslit])
        slitbpm  = reform(bpmask[xmin+j,zslit])
        
        owspec[j] = wavecal[xmin+j,zslit[0]]

;  This is a fix to avoid the edges of the resampled images where
;  often times the bad pixel mask has most of the pixels set as bad.
;  Set values to !values.f_nan and move on.

        junk = where(slitbpm eq 0,cnt)
        if cnt gt 0.25*nslit then begin

           ofspec[j,*] = !values.f_nan
           ouspec[j,*] = !values.f_nan
           print, "Skipping wavelength.  Don't worry, just " + $
                  "a wavelength near the edge of an order."
           continue
           
        endif
        
;  Generate the slit positions using the tracecoeffs

        trace_arc = fltarr(naps,/NOZERO)
        
        for k = 0,naps-1 do begin
           
           l = i*naps+k
           trace_arc[k] = poly(owspec[j],tracecoeffs[*,l])
           
        endfor
        
;  Generate the aperture mask        

        slitmask = mc_mkapmask(slit_arc,trace_arc,replicate(apradius,naps),$
                               PSBGINFO=bginfo,CANCEL=cancel)
        if cancel then return, -1

        if doopt then begin

           psfmask = mc_mkapmask(slit_arc,trace_arc,replicate(psfradius,naps), $
                                 CANCEL=cancel)
           if cancel then return,-1
           
        endif
                       
;  Do the background subtraction if requested
        
        if dobgsub then begin

           zbg = where(slitmask eq -1 and slitvar ne 0.0,cnt)

           if cnt ge bginfo[2]+2 then begin

;  Find outliers including the bad pixels

              mc_moments,slitimg[zbg],mean,vvar,SILENT=1, $
                         IGOODBAD=slitbpm[zbg],OGOODBAD=ogoodbad,ROBUST=5, $
                         CANCEL=cancel
              if cancel then return,-1
              
;  Now fit the background ignoring these pixels

              coeff  = mc_robustpoly1d(slit_arc[zbg],slitimg[zbg],bginfo[2], $
                                       4,0.1,YERR=sqrt(slitvar[zbg]),/SILENT,$
                                       IGOODBAD=ogoodbad,OGOODBAD=ogoodbad, $
                                       COVAR=cvar,CANCEL=cancel)
              if cancel then return,-1
              
;  Debug plotting
           
              if debugbgsub and xmin+j ge debugxrange[0] and $
                 xmin+j le debugxrange[1] then begin
                 
                 min = min(slitimg,MAX=max,/NAN)
                 
                 plot, slit_arc,slitimg,/xsty,/ysty,PSYM=10, $
                       TITLE='BG Sub Window-Column '+strtrim(xmin+j,2),$
                       YRANGE=mc_bufrange([min,max],0.1)
                 plots,slit_arc[zbg],slitimg[zbg],PSYM=2,COLOR=2

                 junk = where(ogoodbad eq 0,cnt)
                 if cnt ne 0 then plots,slit_arc[zbg[junk]],slitimg[zbg[junk]],$
                                        COLOR=3,SYMSIZE=2,PSYM=4
                 oplot,slit_arc,poly(slit_arc,coeff),COLOR=6
                 cancel = mc_pause()
                 if cancel then return, -1
                 
              endif
              
;  Subtract the background
              
              slitimg = temporary(slitimg)- $
                        mc_poly1d(slit_arc,coeff,cvar,YVAR=yvar)
              slitvar = temporary(slitvar)+yvar
              
           endif else begin

              mc_message,'Not enough background points found at column '+$
                         strtrim(xmin+j,2)+', wavelength '+$
                         strtrim(owspec[j],2)+' in order '+ $
                         strtrim(orders[i],2)+'.',WIDGET_ID=widget_id,/ERROR
              cancel = 1
              return, -1

           endelse

                    
        endif

;  Scale the profile model

        if doopt or dobpf then begin

           modprof = mc_sincinterp(spatmod.sgrid,reform(spatmod.model[j,*]), $
                                   slit_arc,CANCEL=cancel)
           if cancel then return,-1

           scoeff = mc_robustpoly1d(modprof,slitimg,1,thresh,0.1,/SILENT,$
                                    OGOODBAD=ogoodbad,IGOODBAD=slitbpm,$
                                    CANCEL=cancel)
           if cancel then return, -1
           
;  Debug plotter

           if debugbpf and xmin+j ge debugxrange[0] and $
              xmin+j lt debugxrange[1] then begin

              sclmodprof = poly(modprof,scoeff)
              zbad = where(ogoodbad eq 0,cnt)
                         
              plot, slit_arc,slitimg,/XSTY,PSYM=10,/YSTY, $
                    TITLE='Bad Pixel Window, Column - '+strtrim(xmin+j,2)+$
                    'Wavelength - '+strtrim(owspec[j],2)
              
              oplot,slit_arc,sclmodprof,PSYM=10,COLOR=2
              if cnt ne 0 then oplot,slit_arc[zbad],slitimg[zbad],COLOR=4, $
                                     PSYM=2,SYMSIZE=2
              cancel = mc_pause()
              if cancel then return, -1
              
           endif
           
        endif 
        
;  Do the extraction

        for k = 0,naps-1 do begin
        
           if doopt then begin

              zpsf = where(psfmask gt float(k) and psfmask le float(k+1),junk)

;  Enforce positivity and normalize.  For ease of indexing, 
              
              posprof = (apsign[k] eq 1) ? (modprof>0.0):(modprof<0.0)
              approf = apsign[k]*abs(modprof/total(posprof,/NAN))

;  Scale data values
              
              zap = where(slitmask gt float(k) and slitmask le float(k+1) and $
                          approf ne 0.0 and ogoodbad eq 1,cnt)
              
              if cnt ne 0 then begin
                 
                 vals  = slitimg[zap]/approf[zap]
                 vals_var = slitvar[zap]/approf[zap]^2

                 
                 
                 mc_meancomb,vals,mean,mvar,DATAVAR=vals_var
                 ofspec[j,k] = mean
                 ouspec[j,k] = sqrt(mvar)

;  Debug plotter
              
                 if debugopt and $
                    xmin+j ge debugxrange[0] and $
                    xmin+j le debugxrange[1] then begin

                    min = min([vals+sqrt(vals_var),vals-sqrt(vals_var)],$
                                 MAX=max)
                    
                    plot,slit_arc[zap],vals,/NODATA, $
                         TITLE='Optimal Extraction Window, Column '+ $
                         strtrim(xmin+j,2)+' Aperture '+ $
                         strtrim(k+1,2),/XSTY,/YSTY, $
                         YRANGE=mc_bufrange([min,max],0.1)
                    
                    oploterr,slit_arc[zap],vals,sqrt(vals_var),6
                    cancel = mc_pause()
                    if cancel then return, -1
                    
                 endif
                 
              endif else begin

                 ofspec[j,k] = 0.0
                 ouspec[j,k] = 1
                 obspec[j,k] = obspec[j,k] + 8
                 print, 'Optimal extraction failed at column '+ $
                        strtrim(xmin+j,2)+', wavelength '+$
                        strtrim(owspec[j],2)+' in order '+ $
                        strtrim(orders[i],2)+'.'
                 
              endelse
              
           endif else begin

;              if xmin+j ge debugxrange[0] and $
;                 xmin+j le debugxrange[1] then begin
;
;                 print, xmin+j
;                 print, slitimg[zap]
;                 print, total(slitimg[zap]*(slitmask[zap]-float(k)))*$
;                    float(apsign[k])
;                 cancel = mc_pause()
;                 if cancel then return,-1
;                 
;              endif
              
;  Do standard extraction

              if dobpf then begin

;  Determine if there are bad pixels within the aperture
                 
                 zbad = where(ogoodbad eq 0 and $
                              slitmask gt float(k) $
                              and slitmask le float(k+1),nbad)
                 
                 if nbad ne 0 then begin
                    
;  Scale the profile and replace the pixels

                    sclmodprof = poly(modprof,scoeff)
                    slitimg[zbad] = sclmodprof[zbad]
                    
;  Now scale the profile to the variange image and replace
                    
                    coeff = mc_robustpoly1d(abs(modprof),slitvar,1,thresh,0.01,$
                                            /SILENT,CANCEL=cancel)
                    sclmodprof = poly(abs(modprof),coeff)
                    slitvar[zbad] = sclmodprof[zbad]
                    
                    obspec[j,k] = obspec[j,k]+2
                    
                 endif

              endif

;  Do the simple sum

              zap = where(slitmask gt float(k) and slitmask le float(k+1))
                 
              ofspec[j,k] = total(slitimg[zap]*(slitmask[zap]-float(k)))*$
                            float(apsign[k])
              
              ouspec[j,k] = sqrt(total(slitvar[zap]*(slitmask[zap]-float(k))^2))

;              print, slitmask
;              plot,slit_arc,slitimg,/XSTY,/YSTY,PSYM=10
;              oplot,slit_arc[zap],slitimg[zap],PSYM=1,COLOR=2
;                           
;              cancel = mc_pause()
;              if cancel then return,-1
              
           endelse

;  Now set the bit mask for linearity

           junk = where(slitbsm eq 1,cnt)
           if cnt ne 0 then obspec[j,k] = obspec[j,k]+1
                     
        endfor

     endfor
     
     if keyword_set(UPDATE) then begin

        if keyword_set(WIDGET_ID) then begin
        
           if cancelbutton then begin
              
              cancel = progressBar->CheckCancel()
              if cancel then begin
                 
                 progressBar->Destroy
                 obj_destroy, progressbar
                 cancel = 1
                 return, -1
                 
              endif
              
           endif
           percent = (i+1)*(100./float(norders))
           progressbar->update, percent
           
        endif

     endif else begin

        if norders gt 1 then mc_loopprogress,i,0,norders-1

     endelse

;  Store the results

     nonan = mc_nantrim(owspec,2,CANCEL=cancel)
     if cancel then return,-1
     
     for k = 0,naps-1 do begin
        
        name = 'ORD'+string(orders[i],FORMAT='(I3.3)')+ $
               'AP'+string(k+1,FORMAT='(I2.2)')
        
        array = [[owspec[nonan]],$
                 [ofspec[nonan,k]],$
                 [ouspec[nonan,k]],$
                 [obspec[nonan,k]]]
        
        struc = (m eq 0) ? $
                create_struct(name,array):create_struct(struc,name,array)
        
        m = m + 1
        
     endfor
            
  endfor
  
  if debugbgsub or debugbpf or debugopt then wdelete, wid
  
  done:

  if keyword_set(WIDGET_ID) then begin
     
     progressbar-> destroy
     obj_destroy, progressbar
     
  endif
  
  return, struc
  
end
