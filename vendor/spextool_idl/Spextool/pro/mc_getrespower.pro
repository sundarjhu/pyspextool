;+
; NAME:
;     mc_getrespower
;
; PURPOSE:
;     To compute the slit-limited resolving power for an array of wavelengths
;
; CALLING SEQUENCE:
;     result = mc_getrespower(waves,slitw_pix,FILLENDS=fillends,
;                             DLAMBDA=lambda,DISPERSION=dispersion,$
;                             CANCEL=cancel)
;
; INPUTS:
;     waves     - An 1D array of wavelengths.
;     slitw_pix - A scalar giving the slit width in pixels.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     DLAMBDA   - A 1D array giving the delta lambda for each wavelength.
;     DISPERSON - A 1D array giving the dispersion at each wavelength.
;     FILLENDS  - Set the edge pixels which are typically NaN to the
;                 resolving of the nearest non-NaN pixel.
;     CANCEL    - Set on return if there is a problem.
;
; OUTPUTS:
;     result - A 1D array giving the resolving power (lambda/delta
;              lambda) at each wavelength.
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
;     The Astronomy User's Library
;
; PROCEDURE:
;     At each wavelength, the delta lambda centered on this wavelength
;     for the given slit width in pixels is computed.  The resolving
;     power is then given by lambda/(delta lambda).  
;
; EXAMPLES:
;     NaN
;
; MODIFICATION HISTORY:
;     2018-March-10:  Written by M. Cushing, University of Toledo
;     2018-May-14: Added the DISPERSION keyword.
;-
function mc_getrespower,waves,slitw_pix,DLAMBDA=dlambda,DISPERSION=dispersion,$
                        FILLENDS=fillends,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() ne 2 then begin
     
     print, 'Syntax - result = mc_getrespower(waves,slitw_pix,DLAMBDA=dlambda,$'
     print, '                                 DISPERSION=dispersion,$'
     print, '                                 FILLENDS=fillends,CANCEL=cancel)'
     cancel = 1
     return,-1
     
  endif
  cancel = mc_cpar('mc_getrespower',waves,1,'Waves',[4,5],1)
  if cancel then return,-1
  cancel = mc_cpar('mc_getrespower',slitw_pix,2,'Slitw_pix',[2,3,4,5],0)
  if cancel then return,-1
    
  ndat = n_elements(waves)
  x = findgen(ndat)
  
  respower = fltarr(ndat)
  dlambda = fltarr(ndat)

  dispersion = waves-shift(waves,1)
  dispersion[0] = dispersion[1]  
  
  for i = 0,ndat-1 do begin

     linterp,x,waves,[i-slitw_pix/2.,i+slitw_pix/2.],vals,MISSING=!values.f_nan
     dlambda[i] = vals[1]-vals[0]
     respower[i] = waves[i]/dlambda[i]
     
  endfor

  if keyword_set(FILLENDS) then begin

     z = where(finite(respower) eq 1,cnt)

     respower[0:(z[0]-1)] = respower[z[0]]
     respower[(z[cnt-1]+1):*] = respower[z[cnt-1]]

     dlambda[0:(z[0]-1)] = dlambda[z[0]]
     dlambda[(z[cnt-1]+1):*] = dlambda[z[cnt-1]]     
     
  endif
  
  return, respower

end
