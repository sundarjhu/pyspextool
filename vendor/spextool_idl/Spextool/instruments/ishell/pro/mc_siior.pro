;+
; NAME:
;     mc_siior
;
; PURPOSE:
;     To return the index of refraction for Si.
;
; CALLING SEQUENCE:
;     result = mc_siior(wave,temp,CANCEL=cancel)
;
; INPUTS:
;     wave - A scalar or array of wavelengths in units of microns.
;     temp - The temperature in K.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;     The index of refraction of Si at the requested wavelength(s).  
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     1) The temperatures must be between 20 and 300 K.
;     2) The wavelengths must be between 1.1 and 5.6 microns.
;
; DEPENDENCIES:
;     None
;
; PROCEDURE:
;     Generates the index of refraction using the formula's in
;     Frey et al. (2006, SPIE, 6273, 2J)
;
; EXAMPLES:
;
;
; MODIFICATION HISTORY:
;     2017-06-19 - Written by M. Cushing, University of Toledo
;-
function mc_siior,lambda,temp,CANCEL=cancel

  cancel = 0

  z = where(lambda lt 1.1 or lambda gt 5.6, cnt)
  if cnt ne 0 then begin

     print, 'Warning:  Values only good for 1.1 < lambda < 5.6 um.'
     
  endif

  if temp lt 20 or temp gt 300 then begin

     print, 'Warning:  Values only good for 20 < temp < 300 K.'
     
  endif

  sc1 = [10.4907D,-2.08020d-4,4.21694d-6,-5.82298d-9,3.44688d-12]
  sc2 = [-1346.61D,29.1664,-0.278724,1.05939d-3,-1.35089d-6]
  sc3 = [4.42827d7,-1.76213d6,-7.61575d4,678.414,103.243]

  lc1 = [0.299713D,-1.14234d-5,1.67134d-7,-2.51049d-10,2.32484d-14]
  lc2 = [-3.51710d3,42.3892,-0.357957,1.17504d-3,-1.13212d-6]
  lc3 = [1.71400d6,-1.44984d5,-6.9074d3,-39.3699,23.5770]

  s = [poly(temp,sc1),poly(temp,sc2),poly(temp,sc3)]
  l = [poly(temp,lc1),poly(temp,lc2),poly(temp,lc3)]

  n21 = s[0]*lambda^2/(lambda^2-l[0]^2)+$
        s[1]*lambda^2/(lambda^2-l[1]^2)+$
        s[2]*lambda^2/(lambda^2-l[2]^2)

  return, sqrt(n21+1)
        
end
