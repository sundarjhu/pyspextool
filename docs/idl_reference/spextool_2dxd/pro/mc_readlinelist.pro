;+
; NAME:
;     mc_readlinelist
;
; PURPOSE:
;     To read a Spextool line list
;
; CALLING SEQUENCE:
;     result = mc_readlinelist(ifile,CANCEL=cancel)
;
; INPUTS:
;     ifile - A Spextool line list with | delimited columns of the:
;             order number, wavelengths, ids, fit windows, fit types,
;             and number of fit terms.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     DISTORTION - If set, an additional column is read in which is as
;                  mask array for whether or not the line should be
;                  used in a distortion solution.  
;     CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;     result = A structure with the following tags:
;              result.order   = order numbers (integer)
;              result.swave   = wavelengths (string)
;              result.id      = line IDs (string)
;              result.wwin    = fit window (in units of swave)
;              result.fittype = fit type (0=gaussian, 1=Lotentzian)
;              result.nterms  = Number of terms (3=basic,
;                               4=basic+constant,5=basic+line)
;              [result.mask]  = 1=use in distortion, 0=do not use.
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
;     Just reading a file.
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;     2019-01-30 - Added the distortion keyword.  For now this is a
;                  hack to deal with NIHTS.  
;-
function mc_readlinelist,ifile,DISTORTION=distortion,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() lt 1 then begin
     
     print, 'Syntax - result = mc_readlinelist(ifile,DISTORTION=distortion,$'
     print, '                                  CANCEL=cancel)'
     cancel = 1
     return,-1
     
  endif
  cancel = mc_cpar('mc_readlinelist',ifile,1,'Ifile',[7],0)
  if cancel then return,-1 


  if keyword_set(DISTORTION) then begin

     readcol,ifile,lorders,lswaves,lids,lwins,lfittypes,lnterms,dmask, $
             COMMENT='#',FORMAT='I,A,A,D,A,I,I',/SILENT,DELIMITER='|'
     
     nlines = n_elements(lorders)
     
     lineinfo = {order:lorders[0],swave:lswaves[0],id:lids[0],wwin:lwins[0],$
                 fittype:lfittypes[0],nterms:lnterms[0],dmask:dmask[0]}
     
     lineinfo = replicate(lineinfo,nlines)
     
     lineinfo.order = lorders
     lineinfo.swave = lswaves
     lineinfo.id    = lids
     lineinfo.wwin   = lwins
     lineinfo.fittype = lfittypes
     lineinfo.nterms = lnterms
     lineinfo.dmask = dmask
     
  endif else begin
     
     readcol,ifile,lorders,lswaves,lids,lwins,lfittypes,lnterms,COMMENT='#', $
             FORMAT='I,A,A,D,A,I',/SILENT,DELIMITER='|'
     
     nlines = n_elements(lorders)
     
     lineinfo = {order:lorders[0],swave:lswaves[0],id:lids[0],wwin:lwins[0],$
                 fittype:lfittypes[0],nterms:lnterms[0]}
     
     lineinfo = replicate(lineinfo,nlines)
     
     lineinfo.order = lorders
     lineinfo.swave = lswaves
     lineinfo.id    = lids
     lineinfo.wwin   = lwins
     lineinfo.fittype = lfittypes
     lineinfo.nterms = lnterms

  endelse
     
  return, lineinfo
  
end
  
