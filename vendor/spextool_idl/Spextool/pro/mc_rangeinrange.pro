;+
; NAME:
;     mc_rangeinrange
;
; PURPOSE:
;     To determine how a range [a,b] relates to another range, [c,d].
;
; CALLING SEQUENCE:
;     result = mc_rangeinrange,range,minmax,VALUE=value,CANCEL=cancel
;
; INPUTS:
;     range  - A 2 element array giving a range.
;     minmax - The range against which to check range. 
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     VALUE - An integer describing the result
;
;     5    -  |       |xmin                     |xmax    |
;
;     4    -          |xmin                     |xmax
;             |     |
;
;     3    -          |xmin                     |xmax
;               |     |
;
;     2    -          |xmin                     |xmax
;                  |     |
;
;     1    -          |xmin                     |xmax
;                     |     |
;
;     0    -          |xmin                     |xmax
;                                |     |
;
;     0    -          |xmin                     |xmax
;                     |                         |
;
;     -1   -          |xmin                     |xmax
;                                         |     |
;
;     -2   -          |xmin                     |xmax
;                                            |     |
;
;     -3   -          |xmin                     |xmax
;                                               |     |
;
;     -4   -          |xmin                     |xmax
;                                                 |     |
; OUTPUTS:
;     result - 1=in range (value=2,1,0,-1,-2) , 0=out of range
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
;      
;
; EXAMPLES:
;     IDL> result = mc_rangeinrange([1,4],[2,6],VALUE=value)
;     IDL> print, result, value
;     IDL> 1 2
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;-
function mc_rangeinrange,range,minmax,VALUE=value,CANCEL=cancel

  cancel = 0

;  Check parameters
  
  if n_params() lt 2 then begin
     
     print, 'Syntax - coeff =  mc_rangeinrange(range,minmax,VALUE=value,$'
     print, '                                  CANCEL)'
     cancel = 1
     return, -1
     
  endif
  cancel = mc_cpar('mc_rangeinrange',range,1,'Range',[2,3,4,5],1)
  if cancel then return,-1  
  cancel = mc_cpar('mc_rangeinrange',minmax,2,'Minmax',[2,3,4,5],1)
  if cancel then return,-1  

;  Do the simple case

  if range[0] lt minmax[0] and range[1] gt minmax[1] then begin

     value = 5
     return, -1

  endif
  
  del1 = minmax[0]-range
  del2 = minmax[1]-range

  !except=0
  sign1 = fix(abs(del1)/del1)
  sign2 = fix(abs(del2)/del2)
  void = check_math()
  
  value = total([sign1,sign2],/INTEGER)

  return, (abs(value) le 2) ? 1:-1

end

  
