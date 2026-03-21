;+
; NAME:
;     mc_indxstruc
;
; PURPOSE:
;     Equivalent of subscripting an array but for structures
;
; CALLING SEQUENCE:
;     result = mc_indxstruc(struc,z,CANCEL=cancel)
;
; INPUTS:
;     struc - A structure
;     z     - A array giving the tag positions to extract
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;     result = effectively struc[z]
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
;     Creates a new structure with the tags requested.
;
; EXAMPLES:
;     IDL> x = {one:1,two:2}
;     IDL> y = mc_idxstruc(x,[1])
;     IDL> print, y
;     IDL> 2
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;-
function mc_indxstruc,struc,z,CANCEL=cancel

  cancel = 0

  if n_params() ne 2 then begin
     
     print, 'Syntax - result = mc_indxstruc(struc,z,CANCEL=cancel)'
     cancel = 1
     return,-1
     
  endif
  cancel = mc_cpar('mc_idxstruc',struc,1,'Struc',8,[0,1])
  if cancel then return,-1
  cancel = mc_cpar('mc_idxstruc',z,2,'Z',[1,2,3],[0,1])
  if cancel then return,-1  
  
  tags = tag_names(struc)
  nstruc = create_struct(tags[z[0]],struc.(z[0]))

  for i = 1,n_elements(z)-1 do begin

     nstruc = create_struct(nstruc,tags[z[i]],struc.(z[i]))
     
  endfor
  
  return, nstruc

end
  
