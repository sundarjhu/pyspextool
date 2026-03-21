;+
; NAME:
;     mc_getspextoolinfo
;
; PURPOSE:
;     To obtain basic spextool information like paths and instrument information
;
; CALLING SEQUENCE:
;     mc_getspextoolinfo,spextoolpath,packagepath,instr,notirtf,version,$
;                        INSTRUMENT=instrument,CANCEL=cancel
;
; INPUTS:
;     None
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     INSTRUMENT - If given, spextool will look for the requested
;                  instrument and ignore the default instrument.
;     CANCEL     - Set on return if there is a problem
;
; OUTPUTS:
;     spextoolpath - The full path to the Spextool package.
;     packagepath  - The full path to the instrument directory,
;                    i.e. spextool+'instrument/****/'
;     instr        - A structure with numerous parameters.
;     notirtf      - Set if the instrument is not from the IRTF.
;     version      - Spextool version number.
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
;     Some path parsing.
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;     2017-10-30 - Added the INSTRUMENT keyword.
;-
pro mc_getspextoolinfo,spextoolpath,packagepath,spextoolkeywords,instrinfo, $
                       notirtf,version,INSTRUMENT=instrument,CANCEL=cancel
  
  cancel = 0

;  Get Spextool path

  spextoolpath = file_dirname(file_dirname( $
                 file_which('spextool_instrument.dat')),/MARK)
  
;  Check to see if the instrument was passed in.
  
  if ~keyword_set(INSTRUMENT) then begin
     
     readcol,filepath('spextool_instrument.dat',ROOT_DIR=spextoolpath, $
                      SUBDIR='data'),instrument,COMMENT='#',FORMAT='A',/SILENT
     
     instrument = instrument[0]
     instrfile = strlowcase(strcompress(instrument,/RE))+'.dat'
     
  endif else begin
     
     lcinstrument = strlowcase(instrument)
     instrfile = lcinstrument+'.dat'
     
  endelse
  
;  Check to make sure the instrument file exists.

  check = file_which(instrfile)

  if check eq '' then begin

     if ~keyword_set(PROGRAM) then program = 'Spextool'
     print
     print, 'Error: '+program+' does not recognize the instrument '+ $
            instrument+'.'
     print
     cancel = 1
     return

  endif

;  Note whether it is spex or not.

  notirtf = (strlowcase(instrument[0]) ne 'spex' and $
             strlowcase(instrument[0]) ne 'uspex' and $
             strlowcase(instrument[0]) ne 'ishell') ? 1:0

;  Get package path.

  packagepath = file_dirname(file_dirname(check),/MARK)
  
;  Get instrument info and default settings

  file = filepath(instrfile,ROOT_DIR=packagepath,SUBDIR='data')
  instrinfo = mc_readinstrfile(file,CANCEL=cancel)
  if cancel then return

;  Get Spextool keywords
  
  readcol,filepath('spextool_keywords.dat',ROOT_DIR=spextoolpath, $
                   SUBDIR='data'),spextoolkeywords,COMMENT='#', $
          FORMAT='A',/SILENT
   
;  Get version number

  readcol,file_which('version.dat'),version,FORMAT='A',/SILENT
  

end
