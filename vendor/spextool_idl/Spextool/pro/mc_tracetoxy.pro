;+
; NAME:
;     mc_tracetoxy
;
; PURPOSE:
;     To convert a tracecoeffs to (x,y) positions on the array.
;
; CALLING SEQUENCE:
;     result = mc_tracetoxy(omask,wavecal,spatcal,tracecoeffs,naps,orders, $
;                           doorders,BRUTE=brute,CANCEL=cancel)
;
; INPUTS:
;     omask       - A 2D array where each pixel is set to its order number.
;     wavecal     - A 2D array where each pixel is set to its wavelength.
;     spatcal     - A 2D array where each pixel is set to its angle on
;                   sky.
;     tracecoeffs - A 2D array [ndeg+1,norders*naps] array of
;                   polynomial coefficients that give the position of
;                   the center of the extration aperture.  The
;                   position of the first aperture in the first order
;                   to be extracted in arcseconds is given by,
;                   pos = poly(wave,arr[*,0]).
;     naps        - Number of apertures
;     orders      - A 1D [norders] array giving the order numbers
;     doorders    - A 1D [norders] array. 1=do it, 0=skip it.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     BRUTE  - Set to do the calculation in the brute force way.
;     CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;     A structure with norders*naps tags.  Each tag is a 2D array
;     [nwave,2] where ix = array[*,0] and iy=array[*,1]
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     There is some question as to the accuracy of the gridata
;     conversion, and so really this program should only be used to
;     generate positions for display purposes.
;
; DEPENDENCIES:
;     Spextool library (and its dependencies)
;
; PROCEDURE:
;     Using the trace coefficients, the program computes the aperture
;     position in arcseconds at each wavelength.  Using gridata and
;     the wavecal and spatcal arrays, the program then converts these
;     aperture positions in (x,y) positions on the array.
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017 - Written by M. Cushing, University of Toledo
;     2018-03-01 - Added the BRUTE keyword.
;-
function mc_tracetoxy,omask,wavecal,spatcal,tracecoeffs,naps,orders, $
                      doorders,BRUTE=brute,CANCEL=cancel
  
  cancel = 0

  if n_params() lt 7 then begin

     print, 'Syntax - result = mc_tracetoxy(omask,wavecal,spatcal,$'
     print, '                               tracecoeffs,naps,orders,$'
     print, '                               doorders,BRUTE=brute,CANCEL=cancel)'
     cancel = 1
     return, -1

  endif
  
  cancel = mc_cpar('mc_tracetoxy',omask,1,'Omask',[2,3,4,5],2)
  if cancel then return,-1
  cancel = mc_cpar('mc_tracetoxy',wavecal,2,'Wavecal',[2,3,4,5],2)
  if cancel then return,-1
  cancel = mc_cpar('mc_tracetoxy',spatcal,3,'Spatcal',[2,3,4,5],2)
  if cancel then return,-1
  cancel = mc_cpar('mc_tracetoxy',tracecoeffs,4,'Tracecoeffs',[2,3,4,5],[1,2])
  if cancel then return,-1
  cancel = mc_cpar('mc_tracetoxy',naps,5,'Naps',[2,3,4,5],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_tracetoxy',orders,6,'Orders',[2,3,4,5],[1])
  if cancel then return,-1
  cancel = mc_cpar('mc_tracetoxy',doorders,7,'doorders',[2,3],[1])
  if cancel then return,-1

;  Get basic information and create useful arrays

  s = size(omask,/DIMEN)
  ncols = s[0]
  nrows = s[1]

  norders = n_elements(orders)

  appos = make_array(naps,norders,/FLOAT,VALUE=!values.f_nan)
  
;  Generate the large arrays where each pixel is set to its column/row number
  
  xx = rebin(indgen(ncols),ncols,nrows)
  yy = rebin(reform(indgen(nrows),1,nrows),ncols,nrows)

  if keyword_set(BRUTE) then begin

;     for i = 0,norders-1 do begin
;
;        if doorders[i] eq 0 then continue
;
;;  Find the pixels associated with the order in question
;        
;        zordr = where(omask eq orders[i])
;        
;;  Get the maximum wavelength and pixels values
;        
;        wmax = max(wavecal[zordr],MIN=wmin)
;        xmax = max(xx[zordr],MIN=xmin)        
;        
;        ndat = floor((xmax-xmin+1)/float(brute))
;        xgrid = lindgen(ndat)*brute+xmin
;
;     endfor






     


  endif else begin
  
;  Start the loop over orders
     
     l = 0
     for i = 0,norders-1 do begin
        
        if doorders[i] eq 0 then continue
        
;  Find the pixels associated with the order in question
        
        zordr = where(omask eq orders[i])
        
;  Get the maximum wavelength and pixels values
        
        wmax = max(wavecal[zordr],MIN=wmin)
        xmax = max(xx[zordr],MIN=xmin)
        
;  Compute a wavelength grid
        
        dw = (wmax-wmin)/(xmax-xmin)
        wave = (findgen(xmax-xmin-1))*dw+wmin
        
;  For each aperture within the order, compute the x,y positions
        
        for j = 0,naps-1 do begin
           
           trace_ap = poly(wave,tracecoeffs[*,l])

;           ix = griddata(wavecal[zordr],spatcal[zordr],xx[zordr],POWER=3,$
;                         /POLY,XOUT=wave,YOUT=trace_ap,MISSING=!values.f_nan)
           
;           iy = griddata(wavecal[zordr],spatcal[zordr],yy[zordr],POWER=3,$
;                         /POLY,XOUT=wave,YOUT=trace_ap,MISSING=!values.f_nan)

           triangulate,wavecal[zordr],spatcal[zordr],tri
           
           ix = griddata(wavecal[zordr],spatcal[zordr],xx[zordr],/LINEAR,$
                         TRI=tri,XOUT=wave,YOUT=trace_ap,MISSING=!values.f_nan)
           
           iy = griddata(wavecal[zordr],spatcal[zordr],yy[zordr],/LINEAR,$
                         TRI=tri,XOUT=wave,YOUT=trace_ap,MISSING=!values.f_nan)
           
           
;  Store the results
           
           array = [[ix[*,0]],[iy[*,0]]]
           name = 'AP'+string(l+1,FORMAT='(I2.2)')
           
           struc = (l eq 0) ? $
                   create_struct(name,array):create_struct(struc,name,array)
           
           l++     
           
        endfor
        
     endfor

  endelse
     
  return, struc
  
 
end
