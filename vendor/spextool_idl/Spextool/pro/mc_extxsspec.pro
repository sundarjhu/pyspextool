;+
; NAME:
;
;
; PURPOSE:
;
;
; CALLING SEQUENCE:
;
;
; INPUTS:
;
;
; OPTIONAL INPUTS:
;
;
; KEYWORD PARAMETERS:
;
;
; OUTPUTS:
;
;
; OPTIONAL OUTPUTS:
;
;
; COMMON BLOCKS:
;
;
; RESTRICTIONS:
;
;
; DEPENDENCIES:
;
;
; PROCEDURE:
;
;
; EXAMPLES:
;
;
; MODIFICATION HISTORY:
;     2019-06-24 - Included a check to avoid edges of the array where
;                  the resampled bad pixel mask goes nearly all badcl
;
;-
function mc_extxsspec,image,var,bitmask,omask,orders,wavecal,spatcal, $
                      tracecoeffs,apradii,BPMASK=bpmask,BGINFO=bginfo, $
                      BPFINFO=bpfinfo,UPDATE=update,WIDGET_ID=widget_id, $
                      CANCEL=cancel
  
  cancel = 0

;  Set up debug variables
  
  debugbgsub = 0
  debugbpf = 0
  debugprofmod = 0
  debugxrange = [120,125]

;  Check to see what we are doing

  dobpf = keyword_set(BPFINFO)
  dobgsub = keyword_SET(BGINFO)
  if keyword_set(BGINFO) then bgreg = bginfo.bgreg
  
;  Get array sizes
  
  s       = size(image,/DIMEN)
  ncols   = s[0]
  nrows   = s[1]

  norders = n_elements(orders)
  naps    = n_elements(apradii)

;  Create new arrays
  
  xx = rebin(indgen(ncols),ncols,nrows)
  yy = rebin(reform(indgen(nrows),1,nrows),ncols,nrows)  

  if ~keyword_set(BPMASK) then bpmask = make_array(ncols,nrows,/BYTE,VALUE=1)
;  ximgtool,bpmask
;  cancel = 1
;  return, -1
  
;  Set up debugging window.
  
  if debugbgsub or debugbpf then begin
     
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

     if dobpf then begin
        
        spatmod = mc_mkspatmodel(imgstruc.(i),(profstruc.(i))[*,1], $
                                 tracecoeffs[*,(i*naps):(i*naps+naps-1)],$
                                 apradii,ATMOS=atmos,AVEPROF=medprof, $
                                 DEBUG=debugprofmod,CANCEL=cancel)
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

        slitmask = mc_mkapmask(slit_arc,trace_arc,apradii,XSBGINFO=bgreg, $
                               CANCEL=cancel)
        if cancel then return, -1

;  Do the background subtraction if requested
        
        if dobgsub then begin

           zbg = where(slitmask eq -1 and slitvar ne 0.0,cnt)

           if cnt ge bginfo.bgdeg+2 then begin

;  Find outliers including the bad pixels

              mc_moments,slitimg[zbg],mean,vvar,SILENT=1, $
                         IGOODBAD=slitbpm[zbg],OGOODBAD=ogoodbad,ROBUST=5, $
                         CANCEL=cancel
              if cancel then begin

;  So this is a fix for an odd situation where the user asks to fit
;  the background at just the bottom or just the top of the slit where
;  the resampling often says the pixels are bad at the edges of the
;  orders (see bpmask).  Set values to !values.f_nan and move on.
;  This bug is sorta caught by the new fix above, but leaving it just
;  in case.

                 
                 ofspec[j,*] = !values.f_nan
                 ouspec[j,*] = !values.f_nan
                 print, 'Skipping this wavelength.'
                 continue

              endif
              
;  Now fit the background ignoring these pixels

              coeff  = mc_robustpoly1d(slit_arc[zbg],slitimg[zbg], $
                                       bginfo.bgdeg,4,0.1, $
                                       YERR=sqrt(slitvar[zbg]),/SILENT,$
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
;           
        endif

;  Scale the profile model

        if dobpf then begin

           modprof = mc_sincinterp(spatmod.sgrid,reform(spatmod.model[j,*]), $
                                   slit_arc,CANCEL=cancel)
           if cancel then return,-1

;           print, slitbpm
           
           scoeff = mc_robustpoly1d(modprof,slitimg,1,thresh,0.01,/SILENT,$
                                    OGOODBAD=ogoodbad,IGOODBAD=slitbpm,$
                                    CANCEL=cancel)
;           print, scoeff
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
           
;  Do standard extraction

           if dobpf then begin
              
;  Determine if there are bad pixels within the aperture
              
              zbad = where(ogoodbad eq 0 and slitmask gt float(k) $
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
           
           ofspec[j,k] = total(slitimg[zap]*(slitmask[zap]-float(k)))
           ouspec[j,k] = sqrt(total(slitvar[zap]*(slitmask[zap]-float(k))^2))

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

  if debugbgsub or debugbpf then wdelete, wid
  
  done:

  if keyword_set(WIDGET_ID) then begin
     
     progressbar-> destroy
     obj_destroy, progressbar
     
  endif
  
  return, struc
  
end
