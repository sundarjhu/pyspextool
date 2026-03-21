;+
; NAME:
;     mc_ldcolortempct
;
; PURPOSE:
;     To construct a color table that moves smoothing from blue to red.
;
; CATEGORY:
;     Plotting
;
; CALLING SEQUENCE:
;     mc_ldcolortempct,ndat,INVERT=invert,CANCEL=cancel
;
; INPUTS:
;     ndat - The number of color table values required.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     INVERT - Set for the colors to run from red to blue.
;     CANCEL  - Set on return if there is a problem.
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
; SIDE EFFECTS:
;     Modifies the current color table.
;
; RESTRICTIONS:
;     Requires Spextool
;
; PROCEDURE:
;     NA
;
; EXAMPLE:
;     IDL> mc_ldcolortempct,10,CANCEL=cancel
;
; MODIFICATION HISTORY:
;     2008-01-28 - Written by M. Cushing, Institute for Astronomy,
;                  University of Hawaii
;-
pro mc_ldcolortempct,ndat,INVERT=invert,CANCEL=cancel

  cancel = 0

  if n_params() ne 1 then begin

     print, 'Syntax - mc_ldcolortempct,ndat,CANCEL=cancel'
     cancel = 1
     return

  endif

  cancel  = mc_cpar('mc_ldcolortempct',ndat,1,'Ndat',[2,3,4,5],0)
  if cancel then return

  r = [findgen(ndat)/(ndat-1.)*255.]
  b = [reverse(r)]
  g = [replicate(30,ndat)]

  if keyword_set(INVERT) then begin

     r = reverse(r)
     b = reverse(b)

  endif

  !except = 0
  tvlct,[0B,255B,byte(r)],[0B,255B,byte(g)],[0B,255B,byte(b)]
  void = check_math()
  
  end
