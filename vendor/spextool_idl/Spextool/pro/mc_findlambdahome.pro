;+
; NAME:
;     mc_findlambdahome
;
; PURPOSE:
;     To determine...
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
;
function homefunc,x

  common rooteqn, constant,temp

  return, (x/mc_siior(x,temp) - constant)

end
;
;===============================================================================
;
function mc_findlambdahome,lambdas,orders,homeorder,ttemp,minwave,maxwave, $
                           CANCEL=cancel
  
  cancel = 0
  
  common rooteqn, constant,temp

  temp = ttemp
  
  nlines = n_elements(lambdas)
  lhome = make_array(nlines,/DOUBLE,VALUE=!values.f_nan)
;  lhome = dblarr(nlines)
  
  nindex = mc_siior(lambdas,temp,CANCEL=cancel)
  if cancel then return,-1

  for i = 0,nlines-1 do begin

     if finite(lambdas[i]) eq 0 then continue
     constant = orders[i]*lambdas[i]/homeorder/nindex[i]
     x = [minwave,maxwave,(minwave+maxwave)/2.]
     test = fx_root(x,'homefunc',/DOUBLE)
     lhome[i] = fx_root(x,'homefunc',/DOUBLE)

  endfor
  
  return, lhome

end
