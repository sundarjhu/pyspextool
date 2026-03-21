;+
; NAME:
;     mc_readwavecal
;
; PURPOSE:
;     To read a Spextool wavecal file.
;
; CALLING SEQUENCE:
;     mc_readwavecal,ifile,wctype,flatname,indices,wavecal,wdisp,$
;                    ROTATE=rotate,CANCEL=cancel
;
; INPUTS:
;     ifile - The name of a Spextool wavecal FITS file.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     ROTATE - If given, the wavecal and spatcal images will be
;              rotated by this value (see IDL ROTATE).
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;     wctype  - A scalar string describing the wavelength calibration
;               method.
;     flatname - A scalar string giving the name of the associated
;                flat field image.
;     indices - A structure with norders tags.  Each tag consists of
;               an [nxgrid,nygrid,2] array.  The [*,*,0] array gives the
;               x coordinates of lines of constant wavelength and
;               spatial coorindate while the [*,*,1] gives the y
;               coordinates.  However, there is more information
;               encoded in each plane that simply the coordinates.
;               For  example, here is how the information is encoded
;               in the first plane.
;
;               wgrid=indices[1:*,1,0]    (wavelengths)
;               xgrid=indices[1:*,0,0]    (columns)
;               sgrid=indices[0,2:*,0]    (spatial coordinates)
;               ix   =indices[1:*,2:*,0]  (indices)
;
;     wavecal - The 2D wavelength calibration image.  Each pixel in an
;               order is set to its wavelength.
;     spatcal - The 2D spatial calibration image.  Each pixel in an
;               order is set to its spatial coordinate.
;     wdisp   - The approximate dispersion in wavelengths per pixel for
;               each order.
;
; OPTIONAL OUTPUTS:
;     All
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     Must be used on a Spextool wavecal FITS file.
;
; DEPENDENCIES:
;     The Spextool package (and its dependencies)
;
; PROCEDURE:
;     Just reading from a FITS file
;
; EXAMPLES:
;     eh
;
; MODIFICATION HISTORY:
;     2009-12-03 - Written by M. Cushing, NASA JPL
;     2014-05-28 - Modified to read new MEF files.
;     2014-08-06 - Added wdisp output.
;     2017-08-14 - Heavily modified to read new iSHELL-type wavecal files.
;-
pro mc_readwavecal,ifile,wctype,flatname,indices,wavecal,spatcal, $
                   xranges,wdisp,wavefmt,spatfmt,ROTATE=rotate,CANCEL=cancel

  cancel = 0
  
  if n_params() lt 1 then begin

     cancel = 1
     return

  endif

  cancel = mc_cpar('mc_readwavecal',ifile,1,'Input Filename',7,0)
  if cancel then return

;  Get header information

  hdr = headfits(ifile,EXTEN=0,/SILENT)

  orders    = long( strsplit( fxpar(hdr,'ORDERS'), ',', /EXTRACT) )
  norders   = fxpar(hdr,'NORDERS')
  wctype    = strtrim(fxpar(hdr,'WCTYPE'),2)
  wavefmt   = strtrim(fxpar(hdr,'WAVEFMT'),2)
  spatfmt   = strtrim(fxpar(hdr,'SPATFMT'),2)
  wdisp     = mc_fxpar(hdr,'DISP*')
  flatname  = strtrim(fxpar(hdr,'FLATNAME'))

;  Get extraction ranges

  xranges = intarr(2,norders)
  for i = 0,norders-1 do begin

     name = 'OR'+string(orders[i],FORMAT='(I3.3)')+'_XR'
     xranges[*,i] = long( strsplit( fxpar(hdr,name), ',', /EXTRACT) )

  endfor
  
;  Get wavecal and spatcal images
  
  wavecal = mrdfits(ifile,1,/SILENT)
  spatcal = mrdfits(ifile,2,/SILENT)

  if n_elements(ROTATE) ne 0 then begin

     wavecal = rotate(temporary(wavecal),rotate)
     spatcal = rotate(temporary(spatcal),rotate)

  endif
  
;  Now get the indices

  for i = 0,norders-1 do begin

     tmp = mrdfits(ifile,3+i,/SILENT)
     
     tag = string(i)
     indices = (i eq 0) ? create_struct(tag,tmp):create_struct(indices,tag,tmp)
     
  endfor

end
