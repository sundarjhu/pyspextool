;+
; NAME:
;     mc_pause
;
; PURPOSE:
;     To pause program and wait for user notification to continue
;
; CALLING SEQUENCE:
;     cancel = mc_pause()
;
; INPUTS:
;     None
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return to cancel.
;
; OUTPUTS:
;     None
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
;     None
;
; PROCEDURE:
;     Stick the program in a for loop and it will pause the loop at
;     each iteration and wait for the user to hit enter at the command
;     line.  Type 'q' and hit return to quit out of the for loop.
;
; EXAMPLES:
;     cancel = mc_pause()
;     if cancel then return
;
; MODIFICATION HISTORY:
;     2017-10-07 - Written by M. Cushing, University of Toledo
;-
function mc_pause

  response = ' '
  read, 'q to quit/return to continue:',response

  cancel = (response eq 'q') ? 1:0

  return, cancel

end
