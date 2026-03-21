;+
; NAME:
;     mc_reorder
;
; PURPOSE:
;     Reorders a list of FITS images to be in A, B order.
;
; CATEGORY:
;     File I/O
;
; CALLING SEQUENCE:
;     result = reorder(files,CANCEL=cancel)
;
; INPUTS:
;     files - A string array of (fullpath) file names
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;    FILES  - An array of the file names in A,B order
;    IDX    - An index array that can be used to sort other string arrays.
;    CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;    Returns a string array of file names in A, B order
;
; OPTIONAL OUTPUTS:
;    None
;
; COMMON BLOCKS:
;    None
;
; SIDE EFFECTS:
;    None
;
; RESTRICTIONS:
;    None
;
; PROCEDURE:
;    Extracts the BEAM keyword of pairs of images and reorders them 
;    to be in A, B order.  Useful since the dither pattern of the IRTF
;    is ABBAABBA.
;
; EXAMPLE:
;
; MODIFICATION HISTORY:
;    2001-01-29 - Written by M. Cushing, Institute for Astronomy, UH
;    2009-03-11 - Renamed mc_reorder
;    2016-10-06 - Added the IDX keyword
;    2017-05-19 - Added MESSAGEINFO keyword.
;-
function mc_reorder,fullpaths,FILES=files,IDX=idx,MESSAGEINFO=messageinfo,$
                    CANCEL=cancel

  cancel  = 0
  
;  Check parameters
  
  if n_params() lt 1 then begin
     
     cancel = 1
     print, 'Syntax - result = mc_reorder(files,FILES=files,CANCEL=cancel)'
     return, -1
     
  endif
  cancel = mc_cpar('mc_reorder',fullpaths,1,'Fullpaths',7,[0,1])
  if cancel then return,-1
  
  nfiles = n_elements(fullpaths)
  
  if nfiles mod 2 ne 0 then begin

     
     
     mess = 'Need an even number of images for '
     if n_elements(MESSAGEINFO) then begin

        mess = mess+messageinfo.obj+'.'
        ok = dialog_message(mess,/ERROR,DIALOG_PARENT=messageinfo.id)
        cancel = 1
        return, -1
        
     endif else begin

        mess = mess+'this program.'
        print, mess
        cancel = 1
        return, -1

     endelse
     
  endif
  files = findgen(nfiles)
  idx   = findgen(nfiles)
  
  for i = 0, (nfiles-1)/2 do begin
     
     Ahdr= headfits(fullpaths[i*2])
     beam = fxpar(Ahdr,'BEAM')
     if strcompress(beam,/re) eq 'B' then begin
        
        tmp              = fullpaths[i*2] 
        fullpaths[i*2]   = fullpaths[i*2+1]
        fullpaths[i*2+1] = tmp
        
        tmp              = files[i*2]
        files[i*2]       = files[i*2+1]
        files[i*2+1]     = tmp

        idx[i*2]         = i*2+1
        idx[i*2+1]       = i*2
        
     endif 
     
  endfor
  
  return, fullpaths
  
end
