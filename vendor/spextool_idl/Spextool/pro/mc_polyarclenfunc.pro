;+
; NAME:
;     mc_polyarclenfunc
;
; PURPOSE:
;     To be passed the qsimp routine to determine arc lengths of polynomials
;
; CALLING SEQUENCE:
;     result = mc_polyarclenfunc(x,COEFF=coeff)   
;
; INPUTS:
;     x - value to be evaluated
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     COEFF - the array of coefficients of the polynomial
;
; OUTPUTS:
;     result - the value compute for the function 
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
;     Explicity the mc_poly1dderiv function in the Spextool packpage
;
; PROCEDURE:
;     NA
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-03-22 - Written by M. Cushing, University of Toledo
;-
function mc_polyarclenfunc,x,COEFF=coeff

  return, sqrt( 1.0 + mc_poly1dderiv(x,coeff)^2)

end
