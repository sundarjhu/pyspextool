;+
; NAME:
;     mc_message
;
; PURPOSE:
;     To display a message using a GUI or to the terminal.
;
; CALLING SEQUENCE:
;     mc_message,message, CANCEL=cancel
;
; INPUTS:
;     message - A string scalar (or array) giving the message.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     WIDGET_ID - The widget_id calling the program.  If given, the
;                 message is given via the dialog_message GUI.
;     CANCEL    - Set on return if there is a problem
;
; OUTPUTS:
;     None
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;      None
;
; RESTRICTIONS:
;      None
;
; DEPENDENCIES:
;      Spextool library (and its dependencies)
;
; PROCEDURE:
;      Simple
;
; EXAMPLES:
;      mc_message, 'Hello world',CANCEL=cancel
;
; MODIFICATION HISTORY:
;      2017-10-08 - Written by M. Cushing, University of Toledo
;-
pro mc_message,message,WIDGET_ID=widget_id,_EXTRA=_extra

  cancel = 0

  if n_params() ne 1 then begin
     
     print, 'Syntax - result = mc_message(message,WIDGET_ID=widget_id,$'
     print, '                             CANCEL=cancel)'
     cancel = 1
     return
     
  endif
  cancel = mc_cpar('mc_message',message,1,'Message',7,[0,1])
  if cancel then return,-1
  
  if keyword_set(WIDGET_ID) then begin

     message = dialog_message(message,DIALOG_PARENT=widget_id,_EXTRA=_extra)

  endif else begin

     print
     print, message

  endelse

end
  
