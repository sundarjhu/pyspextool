;+
; NAME:
;     mc_bufrange
;
; PURPOSE:
;     To buffer a plot range to make it look nicer
;
; CALLING SEQUENCE:
;     result = mc_bufrange(range,buffer,CANCEL=cancel)
;
; INPUTS:
;       range  - A 2-element array [min,max] to be buffered.
;       buffer - the fraction of the difference (max-min) to be added
;                to expand the range.
;
; OPTIONAL INPUTS:
;       None
;
; KEYWORD PARAMETERS:
;       CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;       A 2-element buffered array.
;
; OPTIONAL OUTPUTS:
;       None
;
; COMMON BLOCKS:
;       None
;
; RESTRICTIONS:
;       None
;
; DEPENDENCIES:
;       Requires the Spextool package.
;
; PROCEDURE:
;       Duh
;
; EXAMPLES:
; 
;
; MODIFICATION HISTORY:
;       2016-12-15 - Written by M. Cushing, University of Toledo
;-
function mc_bufrange,range,buffer,CANCEL=cancel

  cancel = 0

  del = float(range[1]-range[0])
  return, [range[0]-del*buffer,range[1]+del*buffer]

end
