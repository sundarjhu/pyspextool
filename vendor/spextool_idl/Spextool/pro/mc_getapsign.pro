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
;
;-
function mc_getapsign,profiles,appos,doorders,CANCEL=cancel

  cancel = 0

  z = where(doorders eq 1,norders)
  naps = (size(appos,/DIMEN))[0]
  
  apsign = intarr(naps,norders)

  for i = 0,norders-1 do begin

     profile = profiles.(z[i])
     x = profile[*,0]
     y = profile[*,1]        
     medy = median(y,/EVEN)

     for j = 0,naps-1 do begin
     
        tabinv, x, appos[j,i], xpix
        apsign[j,i] = (y[xpix] gt medy ) ? 1:-1
        
     endfor
        
  endfor
  
  return, apsign



end
