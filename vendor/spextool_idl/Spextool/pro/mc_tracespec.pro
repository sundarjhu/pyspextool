;+
; NAME:
;     mc_tracespec
;
; PURPOSE:
;     To trace point sources in spectral data
;
; CALLING SEQUENCE:
;     result = mc_tracespec(img,omask,wavecal,spatcal,orders,xranges,appos,$
;                           step,sumap,winthresh,fitorder,WID=wid,$
;                           OPLOT=oplot,CANCEL=cancel)
;
; INPUTS:
;     img       - A 2D image.
;     omask     - A 2D array with each pixel set to its order number.
;     wavecal   - A 2D array with each pixel set to its wavelength.
;     spatcal   - A 2D array with each pixel set to angle on the sky.
;     orders    - An array of order numbers
;     xranges   - An array [2,norders] of column numbers where the
;                 orders are completely on the array
;     appos     - An [naps,norders] array giving guess aperture
;                 positions in units of spatcal
;     step      - The loop step size in pixels
;     sumap     - The window size to median combine together around the
;                 guess position in pixels
;     winthresh - Threshold over which the fit to a given column is
;                 considered bad.  If abs(fity-guessy) gt winthresh
;                 then the fit is considered bad.
;     fitorder  - The polynomial degree to use to determine the trace
;                 coefficients.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     WID    - If given, the trace positions will be plotted in real
;              time.
;     OPLOT  -  An [[npoints],[npoints],[npoints]] array where:
;               array[*,0] gives the x positions of the fits
;               array[*,1] gives the y positions of the fits
;               array[*,2] is a goodbad array for the fits.  
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;     result - A [ndeg+1,norders*naps] array giving trace coefficients
;              for each aperture in each order.  
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     None
;
; DEPENDENCIES:
;     Spextool library (and its dependencies)
;
; PROCEDURE:
;     Loops over each order and fits the slit with a gaussian centered
;     at the appos position.  Then fits the result x and y values
;     with a polynomial of degree fitorder.  
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;-
function mc_tracespec,img,omask,wavecal,spatcal,orders,xranges,appos,step, $
                      sumap,winthresh,fitorder,WID=wid,OPLOT=oplot,CANCEL=cancel
  
  cancel = 0

  !except = 0

  plotline = 0
  plotpoly = 0
  
  s = size(img,/DIMEN)
  ncols = s[0]
  nrows = s[1]

  norders   = n_elements(orders)
  naps      = n_elements(appos[*,0])

  halfsumap = fix(sumap/2.)
  coeff     = dblarr(fitorder+1,naps*norders)

  xx = rebin(indgen(ncols),ncols,nrows)
  yy = indgen(nrows)

  if plotline then begin

     window
     linewid = !d.window

  endif

  if plotpoly then begin

     window
     polywid = !d.window

  endif

  for i =0,norders-1 do begin

     starts  = xranges[0,i]+step-1 
     stops   = xranges[1,i]-step+1 
     
     numstep = fix((stops-starts)/step)+1
     column  = findgen(numstep)*step + starts
     
     peaks_pix = replicate(!values.f_nan,numstep,naps)
     peaks_arc = replicate(!values.f_nan,numstep,naps)
     waves     = make_array(numstep,/DOUBLE,VALUE=!values.f_nan)
     
     for j = 0,numstep-1 do begin

        colz = reform(median(img[(column[j]-halfsumap)>0: $
                                 (column[j]+halfsumap) < (ncols-1),*], $
                             /EVEN,DIMEN=1))

        omaskz = reform(omask[column[j],*])
        wavez  = reform(wavecal[column[j],*])
        
        zslit = where(omaskz eq orders[i],cnt)

        slitz = reform(colz[zslit])
        slits = reform(spatcal[column[j],zslit])
        slity = reform(yy[zslit])

        waves[j] = wavez[zslit[0]]

;  Get guess values

        for k=0,naps-1 do begin

           
           linterp,slits,slitz,appos[k,i],guessz
           guesss = total(appos[k,i])

           fits = mpfitpeak(slits,slitz,a,NTERMS=4, $
                            ESTIMATES=[guessz,guesss,1,0],XTOL=1e-3,MAXITER=10,$
                            NITER=niter)
           
           if abs(a[1]-guesss) le winthresh then begin

              peaks_arc[j,k]=a[1]
              linterp,slits,slity,a[1],val
              peaks_pix[j,k]=val

           endif

           if plotline then begin
              
              wset, linewid
              title = 'Column='+string(column[j],FORMAT='(I4)')+$
                      ', ds='+strtrim(abs(a[1]-guesss),2)+' arcsec, '+$
                      ((finite(peaks_arc[j,k]) eq 1) ? 'Good Fit':'Bad Fit')
                      
              plot, slits,slitz,/XSTY,/YSTY,PSYM=10,TITLE=title,$
                    XTITLE='Slit Position (arcsec)',YTITLE='Relative Intensity'
              
              plots, [guesss,guesss],!y.crange,COLOR=4
              plots, [a[1],a[1]],!y.crange,COLOR=2
              oplot, slits,fits,COLOR=2,PSYM=10
              cancel = mc_pause()
              if cancel then return, -1
                            
           endif

        endfor

        if n_elements(WID) ne 0 then begin
           
           wset, wid                 
           for k = 0, naps-1 do begin

              if finite(a[1])  then begin
                 
                 plots, column[j],peaks_pix[j,k],PSYM=4,COLOR=3
                 
              endif
              
           endfor
           
        endif       
        
     endfor

;  Now fit the objects locations with a polynomial.
     
     for k = 0, naps-1 do begin
        
        l = naps*i+k
       
;  Fit the trace in w/s space

        output = mc_robustpoly1d(waves,peaks_arc[*,k],fitorder,3,0.01, $
                                 OGOODBAD=goodbad,/GAUSSJ,/SILENT,CANCEL=cancel)
        if cancel then return, -1
        
        coeff[*,l] = output

;  Plot it if requested.

        if plotpoly then begin

           wset, polywid
           title = 'Order='+string(orders[i],FORMAT='(I3.3)')+$
                   ', Ap='+string(k+1,FORMAT='(I1)')
           
           plot,[1],[1],TITLE=title,$
                XRANGE=[min(waves,MAX=max,/NAN),max],$
                YRANGE=[min(peaks_arc[*,0],MAX=max),max],$
;                YRANGE=[floor(min(peaks_arc,MAX=max,/NAN)),ceil(max)],$
                /NODATA,$
                /XSTY,/YSTY
           oplot,waves,peaks_arc[*,0],PSYM=4           
           oplot,waves,poly(waves,output),COLOR=2
           bad = where(goodbad eq 0,cnt)
           if cnt ne 0 then oplot,waves[bad],(peaks_arc[*,k])[bad], $
                                  PSYM=4,COLOR=4

        endif

;  Store data for ximgtool

        if l eq 0 then begin

           xoplot = column
           yoplot = peaks_pix[*,k]
           goplot = goodbad

        endif else begin

           xoplot = [xoplot,column]
           yoplot = [yoplot,peaks_pix[*,k]]
           goplot = [goplot,goodbad]

        endelse

     endfor

     if plotpoly then begin

        cancel = mc_pause()
        if cancel then return, -1
        
     endif
     
  endfor

  oplot = [[xoplot],[yoplot],[goplot]]
  
  junk = check_math()
  !except = 1

  return, coeff
  
end

