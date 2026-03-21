;+
; NAME:
;     mc_readinstrfile
;
; PURPOSE:
;     Reads a Spextool instrument calibration file.
;
; CATEGORY:
;     Spectroscopy
;
; CALLING SEQUENCE:
;     result = mc_readinstrfile(filename,CANCEL=cancel)
;
; INPUTS:
;     filename - The name of a Spextool instrument calibration file.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;     Later
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     Must be a Spextool instrument calibration file
;
; PROCEDURE:
;     Easy
;
; EXAMPLE:
;
; MODIFICATION HISTORY:
;     2001-05-10 - Written by M. Cushing, Institute for Astronomy, UH
;     2003-03-27 - Added slowcnt input
;     2003-04-10 - Added readtime input
;     2007-08-xx - Huge rewrite.
;     2007-09-14 - Removed gain and readnoise outputs.
;     2008-02-28 - Added plotwinsize input.
;     2008-08-01 - Added flatmodule input.
;     2010-08-17 - Converted to a function.
;     2011-06-18 - Removed the combimgsub variable.
;     2014-06-18 - Added the possibility to freeze certain widgets.
;     2014-08-06 - Added the ampcorrect parameter.
;     2015-01-03 - Removed the plotsaturation parameter.
;     2015-01-04 - Changed saturation to lincormax.
;     2017-01-01 - Added AUTOXCORR and PLOTXCORR
;     2017-09-17 - Rewrote to be more intelligent
;     2017-10-03 - Removed instrument specific parameters
;-
function mc_readinstrfile,filename,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() lt 1 then begin
    
     print, 'Syntax - result = mc_readinstfile(filename,CANCEL=cancel)'
     cancel = 1
     return,-1
     
  endif
  cancel = mc_cpar('mc_readinstrfile',filename,1,'Filename',7,0)
  if cancel then return,-1
  
  readcol,filename,key,val,COMMENT='#',DELIMITER='=',FORMAT='A,A',/SILENT
  key = strtrim(key,2)
  
;  INSTRUMENT

  z = where(key eq 'INSTRUMENT',cnt)
  if cnt ne 0 then begin

     str = {instrument:strjoin(strtrim(val[z],2))}

  endif else begin
     
     print, 'mc_readinstrfile:  INSTRUMENT not found.'
     cancel = 1
     return, -1

  endelse

;  NCOLS

  z = where(key eq 'NCOLS',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'ncols',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  NCOLS not found.'
     cancel = 1
     return, -1

  endelse

;  NROWS

  z = where(key eq 'NROWS',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'nrows',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  NROWS not found.'
     cancel = 1
     return, -1

  endelse

;  STDIMAGE

  z = where(key eq 'STDIMAGE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'stdimage',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  STDIMAGE not found.'
     cancel = 1
     return, -1

  endelse

;  PLOTWINSIZE

  z = where(key eq 'PLOTWINSIZE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'plotwinsize',float(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  PLOTWINSIZE not found.'
     cancel = 1
     return, -1

  endelse

;  NINT

  z = where(key eq 'NINT',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'nint',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  NINT not found.'
     cancel = 1
     return, -1

  endelse

;  NSUFFIX

  z = where(key eq 'NSUFFIX',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'nsuffix',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  NSUFFIX not found.'
     cancel = 1
     return, -1

  endelse
  
;  BADPIXELMASK

  z = where(key eq 'BADPIXMASK',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'bdpxmk',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  BADPIXELMASK not found.'
     cancel = 1
     return, -1

  endelse

;  CALMODULE

  z = where(key eq 'CALMODULE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'calmodule',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  CALMODULE not found.'
     cancel = 1
     return, -1

  endelse

;  FILEREADMODE

  z = where(key eq 'FILEREADMODE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'filereadmode',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  FILEREADMODE not found.'
     cancel = 1
     return, -1

  endelse

;  OPREFIX

  z = where(key eq 'OPREFIX',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'oprefix',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  OPREFIX not found.'
     cancel = 1
     return, -1

  endelse

;  SUFFIX

  z = where(key eq 'SUFFIX',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'suffix',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  SUFFIX not found.'
     cancel = 1
     return, -1

  endelse

;  FITSREADPROGRAM

  z = where(key eq 'FITSREADPROGRAM',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'fitsreadprogram',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  FITSREADPRGRAM not found.'
     cancel = 1
     return, -1

  endelse

;  REDUCTION MODE

  z = where(key eq 'REDUCTIONMODE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'reductionmode',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  REDUCTIONMODE not found.'
     cancel = 1
     return, -1

  endelse

;  COMBMODE

  z = where(key eq 'COMBMODE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'combimgmode',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  COMBMODE not found.'
     cancel = 1
     return, -1

  endelse

;  COMBSTAT

  z = where(key eq 'COMBSTAT',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'combimgstat',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  COMBSTAT not found.'
     cancel = 1
     return, -1

  endelse            

;  COMBTHRESH

  z = where(key eq 'COMBTHRESH',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'combimgthresh',strjoin(strtrim(val[z],2)))  

  endif else begin
     
     print, 'mc_readinstrfile:  COMBTHRESH not found.'
     cancel = 1
     return, -1

  endelse
  
;  COMBODIR

  z = where(key eq 'COMBODIR',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'combimgdir',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  COMBDIR not found.'
     cancel = 1
     return, -1

  endelse

;  PSNAPS

  z = where(key eq 'PSNAPS',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'psnaps',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  PSNAPS not found.'
     cancel = 1
     return, -1

  endelse

;  PSPSFRAD

  z = where(key eq 'PSPSFRAD',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'pspsfrad',strjoin(strtrim(val[z],2)))     

  endif else begin
     
     print, 'mc_readinstrfile:  PSPSFRAD not found.'
     cancel = 1
     return, -1

  endelse

;  PSAPRAD

  z = where(key eq 'PSAPRAD',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'psaprad',strjoin(strtrim(val[z]),2))     

  endif else begin
     
     print, 'mc_readinstrfile:  PSAPRAD not found.'
     cancel = 1
     return, -1

  endelse  

;  PSBGSUB

  z = where(key eq 'PSBGSUB',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'psbgsub',fix(strtrim(val[z])))     

  endif else begin
     
     print, 'mc_readinstrfile:  PSBGSUB not found.'
     cancel = 1
     return, -1

  endelse    

;  PSBGSTART

  z = where(key eq 'PSBGSTART',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'psbgstart',strjoin(strtrim(val[z],2)))     

  endif else begin
     
     print, 'mc_readinstrfile:  PSBGSTART not found.'
     cancel = 1
     return, -1

  endelse  
  
;  PSBGWIDTH

  z = where(key eq 'PSBGWIDTH',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'psbgwidth',strjoin(strtrim(val[z],2)))     

  endif else begin
     
     print, 'mc_readinstrfile:  PSBGWIDTH not found.'
     cancel = 1
     return, -1

  endelse

;  PSBGDEG

  z = where(key eq 'PSBGDEG',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'psbgdeg',strjoin(strtrim(val[z],2)))     

  endif else begin
     
     print, 'mc_readinstrfile:  PSBGDEG not found.'
     cancel = 1
     return, -1

  endelse    


;  XSBGSUB

  z = where(key eq 'XSBGSUB',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'xsbgsub',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  XSBGSUB not found.'
     cancel = 1
     return, -1

  endelse

;  XSBGREG

  z = where(key eq 'XSBGREG',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'xsbgreg',strjoin(strtrim(val[z],2)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  XSBG not found.'
     cancel = 1
     return, -1

  endelse
  
;  XSBGDEG

  z = where(key eq 'XSBGDEG',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'xsbgdeg',strjoin(strtrim(val[z],2)))     

  endif else begin
     
     print, 'mc_readinstrfile:  XSBGDEG not found.'
     cancel = 1
     return, -1

  endelse

;  TRACEDEG

  z = where(key eq 'TRACEDEG',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'tracedeg',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  TRACEDEG not found.'
     cancel = 1
     return, -1

  endelse        

;  TRACESTEP

  z = where(key eq 'TRACESTEP',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'tracestepsize',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  TRACESTEP not found.'
     cancel = 1
     return, -1

  endelse

;  TRACESUMAP

  z = where(key eq 'TRACESUMAP',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'tracesumap',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  TRACESUMAP not found.'
     cancel = 1
     return, -1

  endelse

;  TRACESIGTHRESH

  z = where(key eq 'TRACESIGTHRESH',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'tracesigthresh',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  TRACESIGTHRESH not found.'
     cancel = 1
     return, -1

  endelse                

;  TRACEWINTHRESH

  z = where(key eq 'TRACEWINTHRESH',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'tracewinthresh',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  TRACEWINTHRESH not found.'
     cancel = 1
     return, -1

  endelse

;  BADPIXELTHRESH

  z = where(key eq 'BADPIXELTHRESH',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'bdpxthresh',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  BADPIXELTHRESH not found.'
     cancel = 1
     return, -1

  endelse

;  LINCORMAX

  z = where(key eq 'LINCORMAX',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'lincormax',total(long(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readinstrfile:  LINCORMAX not found.'
     cancel = 1
     return, -1

  endelse

;  PLOTXCORR

  z = where(key eq 'PLOTXCORR',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'plotxcorr',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  LINCORMAX not found.'
     cancel = 1
     return, -1

  endelse

;  RECTMETHOD

  z = where(key eq 'RECTMETHOD',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'rectmethod',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  RECTMETHOD not found.'
     cancel = 1
     return, -1

  endelse                          

;  AMPCOR

  z = where(key eq 'AMPCOR',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'ampcorrect',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  AMPCOR not found.'
     cancel = 1
     return, -1

  endelse

;  LINCOR

  z = where(key eq 'LINCOR',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'lincorrect',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  AMPCOR not found.'
     cancel = 1
     return, -1

  endelse

;  FLATFIELD

  z = where(key eq 'FLATFIELD',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'flatfield',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  FLATFIELD not found.'
     cancel = 1
     return, -1

  endelse    

;  FIXBADPIXELS

  z = where(key eq 'FIXBADPIXELS',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'fixbdpx',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  FIXBADPIXELS not found.'
     cancel = 1
     return, -1

  endelse    

;  OPTEXTRACT

  z = where(key eq 'OPTEXTRACT',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'optextract',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  OPTEXTRACT not found.'
     cancel = 1
     return, -1

  endelse

;  AVEPROF

  z = where(key eq 'AVEPROF',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'aveprof',fix(strsplit(val[z],' ',/EXTRACT)))
     
  endif else begin
     
     print, 'mc_readinstrfile:  AVEPROF not found.'
     cancel = 1
     return, -1

  endelse

;  SPEXTOOL_KEYWORD

  z = where(key eq 'XSPEXTOOL_KEYWORD',cnt)
  if cnt eq 0 then begin

     print, 'mc_readinstrfile:  SPEXTOOL_KEYWORD not found.'
     cancel = 1
     return, -1

  endif else begin

     keywords = strarr(cnt)
     
     for i = 0,cnt-1 do keywords[i] = strjoin(strtrim(val[z[i]],2))

     str = create_struct(str,'xspextool_keywords',keywords)
     
  endelse

;  XCOMBSPEC_KEYWORD

  z = where(key eq 'XCOMBSPEC_KEYWORD',cnt)
  if cnt eq 0 then begin

     print, 'mc_readinstrfile:  XCOMBSPEC_KEYWORD not found.'
     cancel = 1
     return, -1

  endif else begin

     keywords = strarr(cnt)
     
     for i = 0,cnt-1 do keywords[i] = strjoin(strtrim(val[z[i]],2))

     str = create_struct(str,'xcombspec_keywords',keywords)
     
  endelse

;  XTELLCOR_KEYWORD

  z = where(key eq 'XTELLCOR_KEYWORD',cnt)
  if cnt eq 0 then begin

     print, 'mc_readinstrfile:  XTELLCOR_KEYWORD not found.'
     cancel = 1
     return, -1

  endif else begin

     keywords = strarr(cnt)
     
     for i = 0,cnt-1 do keywords[i] = strjoin(strtrim(val[z[i]],2))

     str = create_struct(str,'xtellcor_keywords',keywords)
     
  endelse    

  return, str


end
