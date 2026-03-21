;+
; NAME:
;     mc_readishellfits
;
; PURPOSE:
;     Reads iSHELL FITS images.
;
; CATEGORY:
;     File I/O
;
; CALLING SEQUENCE:
;     mc_readishellfits,files,data,[hdrinfo],[var],KEYWORDS=keywords,$
;                       PAIR=PAIR,ROTATE=rotate,AMPCOR=ampcor,LINCORR=lincorr,$
;                       BITINFO=bitinfo,BITMASK=bitmask,NIMAGES=nimages,$
;                       _EXTRA=_extra,WIDGET_ID=widget_id,CANCEL=cancel
;
; INPUTS:
;     files - A string of (fullpath) file names.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     KEYWORDS  - A string array of keywords to extract from the hdrs. 
;     PAIR      - Set to pair subtract.  Must be even number of input files.
;     ROTATE    - Set to the desired IDL rotation command (ROTATE).
;     AMPCOR    - Set to correct the various amps levels.
;     LINCORR   - Set to correct each image for non-linearity
;     BITINFO   - A structure giving various values with which to
;                 create a bitmask.  In this case, the values should
;                 be: {lincormax:lincormax,lincormaxbit:lincormaxbit}.  
;     BITMASK   - If BITINFO is given, a bit-set array according to
;                 the BITINFO values.
;     NIMAGES   - The number of images in the output data
;     WIDGET_ID - If given, an pop-up error message will appear over
;                 the widget.
;     CANCEL    - Will be set on return if there is a problem
;
; OUTPUTS:
;     Returns a floating array of images
;
; OPTIONAL OUTPUTS:
;     hdrinfo - If KEYWORDS are given, an array of structures of the 
;               requested hdr keywords for each image.  If no
;               keywords are given, then all the keywords are
;               returned in a structure.
;     var     - A floating array of variance images is returned
;
; COMMON BLOCKS:
;     None
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     Images are read in, pair subtracted if requested, and
;     variances images are computed if requested.
;
; EXAMPLE:
;     None
;
; MODIFICATION HISTORY:
;     2016-02-04 - Written by M. Cushing, U Toledo.   Based on
;-
function getishellhdrinfo,hdr,KEYWORDS=keywords,CANCEL=cancel

  cancel = 0

;  Get standard header info

  if keyword_set(KEYWORDS) then begin

     hdrinfo = mc_gethdrinfo(hdr,keywords,CANCEL=cancel)
     if cancel then return,-1

     vals = hdrinfo.vals
     coms = hdrinfo.coms

;  Now create the few required keywords for Spextool

     value = fxpar(hdr,'TCS_AM')
     vals = create_struct('AM',value,vals)
     coms = create_struct('AM',' Airmass',coms)              
        
  endif else begin

     value = fxpar(hdr,'TCS_AM')
     vals = create_struct('AM',value)
     coms = create_struct('AM',' Airmass')                   
     
  endelse

  value = fxpar(hdr,'TCS_HA')
  pm = (strmid(value,0,1) eq ' ') ? '+':''
  value = pm+strtrim(value,2)
 
  vals = create_struct('HA',value,vals)
  coms = create_struct('HA',' Hour angle (hours)',coms)

  value = fxpar(hdr,'POSANGLE')
  vals = create_struct('PA',value,vals)
  coms = create_struct('PA',' Position angle E of N (deg)',coms)

  value = fxpar(hdr,'TCS_DEC')

  pm = (strmid(value,0,1) eq ' ') ? '+':''
  value = pm+strtrim(value,2)

  vals = create_struct('DEC',value,vals)
  coms = create_struct('DEC',' Declination, FK5 J2000',coms)  

  value = fxpar(hdr,'TCS_RA')
  vals = create_struct('RA',strtrim(value,2),vals)
  coms = create_struct('RA',' Right Ascension, FK5 J2000',coms)    

  value = fxpar(hdr,'CO_ADDS')
  itime = fxpar(hdr,'ITIME')
  vals = create_struct('IMGITIME',value*double(itime),vals)
  coms = create_struct('IMGITIME', $
                       ' Image integration time, NCOADDSxITIME (sec)',coms)

  vals = create_struct('ITIME',double(itime),vals)
  coms = create_struct('ITIME',' Integration time (sec)',coms)

  vals = create_struct('NCOADDS',value,vals)
  coms = create_struct('NCOADDS',' Number of coadds',coms)  
  
  value = fxpar(hdr,'MJD_OBS')
  vals = create_struct('MJD',string(value,FORMAT='(D16.10)'),vals)
  coms = create_struct('MJD',' Modified Julian date OBSDATE+TIME_OBS',coms)

  value = fxpar(hdr,'TIME_OBS')
  vals = create_struct('TIME',value,vals)
  coms = create_struct('TIME',' Observation time in UTC',coms)

  value = fxpar(hdr,'DATE_OBS')
  vals = create_struct('DATE',value,vals)
  coms = create_struct('DATE',' Observation date in UTC',coms)
  
  value = fxpar(hdr,'IRAFNAME')
  vals = create_struct('FILENAME',value,vals)
  coms = create_struct('FILENAME',' Filename',coms)
  
  value = fxpar(hdr,'XDTILT')
  vals = create_struct('MODE',value,vals)
  coms = create_struct('MODE',' Instrument mode',coms)  
  
  vals = create_struct('INSTR','iSHELL',vals)
  coms = create_struct('INSTR',' Instrument',coms)
  
  return, {vals:vals,coms:coms}
  
end
;
;===============================================================================
;
pro mc_readishellfits,files,data,hdrinfo,var,KEYWORDS=keywords,PAIR=PAIR, $
                      ROTATE=rotate,LINCORR=lincorr,BITINFO=bitinfo,$
                      BITMASK=bitmask,NIMAGES=nimages,_EXTRA=extra, $
                      WIDGET_ID=widget_id,CANCEL=cancel
  
  cancel  = 0

;  Check parameters

  if n_params() lt 1 then begin
     
     cancel = 1
     print, 'Syntax - mc_readishellfits,files,data,[hdrinfo],[var],$'
     print, '                           KEYWORDS=keywords,PAIR=PAIR,$'
     print, '                           ROTATE=rotate,LINCORR=lincorr,$'
     print, '                           BITINFO=bitinfo,BITMASK=bitmask,$'
     print, '                           NIMAGES=nimages,_EXTRA=_extra,$'
     print, '                           WIDGET_ID=widget_id,CANCEL=cancel'
     return
     
  endif

  cancel = mc_cpar('mc_readishellfits',files,1,'Files',7,[0,1])
  if cancel then return
  
;  Get setup info.
  
  dovar    = (arg_present(var) eq 1) ? 1:0
  dolinmax = (n_elements(BITINFO) ne 0) ? 1:0
  rot      = (n_elements(ROTATE) ne 0) ? rotate:0

  readnoise = 10.0  ;  per single read
  gain = 1.8        ;  electrons per DN
  
;  Correct for non-linearity?

  if keyword_set(LINCORR) then begin

     lcfile = filepath('ishell_lincorr_CDS.fits', $
                       ROOT=file_dirname(file_which('ishell.dat'),/MARK))
     
     lc_coeffs = readfits(lcfile,/SILENT)
     
  endif

;  Checking for linearity maximum?

  if dolinmax then begin
  
     biasfile = filepath('ishell_bias.fits', $
                         ROOT=file_dirname(file_which('ishell.dat'),/MARK))
     bias = readfits(biasfile,/SILENT,hdr)
     bias = float(temporary(bias))/fxpar(hdr,'DIVISOR')

  endif
  
;  Correct for amplification offsets?

  doampcor = (n_elements(extra) ne 0) ? extra.test:0

;  Get number of images and check to make sure even number if /PAIR.

  nfiles = n_elements(files)

  if keyword_set(PAIR) then begin
     
     result = mc_crange(nfiles,0,'Number of Files',/EVEN,WIDGET_ID=widget_id,$
                     CANCEL=cancel)
     if cancel then return
     nimages = nfiles/2
     
  endif else nimages = nfiles
  
;  Make data arrays.

  NAXIS1 = 2048
  NAXIS2 = 2048

  data    = fltarr(NAXIS1,NAXIS2,nimages)
  hdrinfo = replicate(getishellhdrinfo(headfits(files[0]),KEYWORDS=keywords), $
                      nfiles)
  
  if dovar then var  = fltarr(NAXIS1,NAXIS2,nimages)
  bitmask = bytarr(NAXIS1,NAXIS2,nimages)
  
  if keyword_set(PAIR) then begin
     
     for i = 0, nimages-1 do begin

;  Get header info for A image
        
        Ahdr   = headfits(files[i*2],EXTEN=0,/SILENT)

        AITIME    = float(fxpar(Ahdr,'ITIME'))
        ACOADDS   = float(fxpar(Ahdr,'CO_ADDS'))
        ANDRS     = float(fxpar(Ahdr,'NDR'))
        AREADTIME = fxpar(Ahdr,'TABLE_SE')
        ADIVISOR  = float(fxpar(Ahdr,'DIVISOR'))

;  Get set up for the error propagation and store total exposure time

        ardvar   = (2.*readnoise^2)/ANDRS/ACOADDS/AITIME^2/gain^2
        acrtn    = (1.0 - AREADTIME*(ANDRS^2 -1.0)/3./AITIME/ANDRS)

;  Read images, get into units of DN.

        Aimg_P = readfits(files[i*2],EXTEN=1,/SILENT)/ADIVISOR
        Aimg_S = readfits(files[i*2],EXTEN=2,/SILENT)/ADIVISOR

;  Check for saturation
        
        if dolinmax then begin
                     
           mask_P = (Aimg_P lt (bias - double(bitinfo.lincormax)))* $
                    2^(bitinfo.lincormaxbit)

           mask_S = (Aimg_S lt (bias - double(bitinfo.lincormax)))* $
                    2^(bitinfo.lincormaxbit)

           Amask = mc_combflagstack([[[mask_P]],[[mask_S]]], $
                                    bitinfo.lincormaxbit+1,CANCEL=cancel)
           
        endif 
        
;  Creat real image
        
        Aimg = Aimg_P-Aimg_S

;  Correct for amplifier offsets

        if doampcor then begin

           Aimg = mc_ishellampcor(Aimg,SILENT=1,CANCEL=cancel)
           if cancel then return
                      
        endif
            
;  Determine the correction for the image

        if keyword_set(LINCORR) then begin

           Acor = mc_imgpoly(Aimg,reform(lc_coeffs[*,*,2:*]),CANCEL=cancel)
           if cancel then return

;  Check for pixels above and below the fit limits.

           msk = Aimg lt reform(lc_coeffs[*,*,0])
           z = where(msk eq 1,cnt)
           if cnt ne 0 then Acor[z] = 1.0

           msk = Aimg gt reform(lc_coeffs[*,*,1])
           z = where(msk eq 1,cnt)
           if cnt ne 0 then Acor[z] = 1.0
           
;  Check for pixels above lincormax and set correction to unity.

           z = where(Amask eq 2^bitinfo.lincormaxbit,cnt)
           if cnt ne 0 then Acor[z] = 1.0
           
;  Set black pixel corrections to unity as well.

           Acor[0:3,*] = 1.0
           Acor[2044:2047,*] = 1.0
           Acor[*,0:3] = 1.0
           Acor[*,2044:2047] = 1.0
           
;  Apply the corrections

           Aimg = temporary(Aimg)/Acor

           delvarx,Acor,Aimg_S, Aimg_P

        endif

;  Get header info for B image

        Bhdr   = headfits(files[i*2+1],EXTEN=0,/SILENT)

        BITIME    = float(fxpar(Bhdr,'ITIME'))
        BCOADDS   = float(fxpar(Bhdr,'CO_ADDS'))
        BNDRS     = float(fxpar(Bhdr,'NDR'))
        BREADTIME = fxpar(Bhdr,'TABLE_SE')
        BDIVISOR  = float(fxpar(Bhdr,'DIVISOR'))

;  Get set up for the error propagation and store total exposure time

        brdvar   = (2.*readnoise^2)/BNDRS/BCOADDS/BITIME^2/gain^2
        bcrtn    = (1.0 - BREADTIME*(BNDRS^2 -1.0)/3./BITIME/BNDRS)
        
;  Read images, get into units of DN.

        Bimg_P = readfits(files[i*2+1],EXTEN=1,/SILENT)/BDIVISOR
        Bimg_S = readfits(files[i*2+1],EXTEN=2,/SILENT)/BDIVISOR

;  Check for saturation
        
        if dolinmax then begin

           mask_P = (Bimg_P lt (bias - double(bitinfo.lincormax)))* $
                    2^(bitinfo.lincormaxbit)

           mask_S = (Bimg_S lt (bias - double(bitinfo.lincormax)))* $
                    2^(bitinfo.lincormaxbit)

           Bmask = mc_combflagstack([[[mask_P]],[[mask_S]]], $
                                    bitinfo.lincormaxbit+1,CANCEL=cancel)
           
        endif 
        
;  Creat real image
        
        Bimg = Bimg_P-Bimg_S

;  Correct for amplifier offsets

        if doampcor then begin

           Bimg = mc_ishellampcor(Bimg,SILENT=1,CANCEL=cancel)
           if cancel then return
                      
        endif
        
;  Determine the correction for the image

        if keyword_set(LINCORR) then begin

           Bcor = mc_imgpoly(Bimg,reform(lc_coeffs[*,*,2:*]),CANCEL=cancel)
           if cancel then return

;  Check for pixels above and below the fit limits.

           msk = Bimg lt reform(lc_coeffs[*,*,0])
           z = where(msk eq 1,cnt)
           if cnt ne 0 then Bcor[z] = 1.0

           msk = Bimg gt reform(lc_coeffs[*,*,1])
           z = where(msk eq 1,cnt)
           if cnt ne 0 then Bcor[z] = 1.0
           
;  Check for pixels above lincormax and set correction to unity.
           
           z = where(Bimg gt double(bitinfo.lincormax),cnt)
           if cnt ne 0 then Bcor[z] = 1.0

;  Set black pixel corrections to unity as well.

           Bcor[0:3,*] = 1.0
           Bcor[2044:2047,*] = 1.0
           Bcor[*,0:3] = 1.0
           Bcor[*,2044:2047] = 1.0
           
;  Apply the corrections

           Bimg = temporary(Bimg)/Bcor

           delvarx,Bcor,Bimg_S,Bimg_P

        endif

;  Combine saturation masks

        if dolinmax then begin

           mask = mc_combflagstack([[[Amask]],[[Bmask]]], $
                                   bitinfo.lincormaxbit+1,CANCEL=cancel)
           if cancel then return
           bitmask[*,*,i] = rotate(mask,rot)
           
        endif 

;  Covert image back to total DN for error propagation and rotate image

        Aimg = temporary(Aimg)*ADIVISOR
        Bimg = temporary(Bimg)*BDIVISOR

        if dovar then begin

           Avar = abs(Aimg)*acrtn/ANDRS/(ACOADDS^2)/(AITIME^2)/gain + ardvar
           Bvar = abs(Bimg)*bcrtn/BNDRS/(BCOADDS^2)/(BITIME^2)/gain + brdvar

           variance = Avar+Bvar
           delvarx,AVar,Bvar

        endif

        img = (Aimg/ADIVISOR/AITIME-Bimg/BDIVISOR/BITIME)
        delvarx,Aimg,Bimg

;  Rotate, and store data and variance if requested


        data[*,*,i] = rotate(img,rot)
        if dovar then var[*,*,i] = rotate(variance,rot)
        delvarx,img,variance

;  Store header information

        copy_struct_inx,getishellhdrinfo(Ahdr,KEYWORDS=keywords),hdrinfo, $
                        index_to=i*2
        copy_struct_inx,getishellhdrinfo(Bhdr,KEYWORDS=keywords),hdrinfo, $
                        index_to=i*2+1
        
     endfor
     
  endif
  
  if not keyword_set(PAIR) then begin
     
     for i = 0, nimages-1 do begin

;  Get hdr info

        hdr = headfits(files[i],EXTEN=0,/SILENT)

        ITIME    = float(fxpar(hdr,'ITIME'))
        COADDS   = float(fxpar(hdr,'CO_ADDS'))
        NDRS     = float(fxpar(hdr,'NDR'))
        READTIME = fxpar(hdr,'TABLE_SE')
        DIVISOR  = float(fxpar(hdr,'DIVISOR'))

;  Get set up for error propagation and store total exposure time

        rdvar   = (2.*readnoise^2)/NDRS/COADDS/ITIME^2/gain^2
        crtn    = (1.0 - READTIME*(NDRS^2 -1.0)/3./ITIME/NDRS)

;  Read images, get into units of DN.

        img_P = readfits(files[i],EXTEN=1,/SILENT)/DIVISOR
        img_S = readfits(files[i],EXTEN=2,/SILENT)/DIVISOR

;  Check for saturation

        if dolinmax then begin

           mask_P = (img_P lt (bias - double(bitinfo.lincormax)))* $
                    2^(bitinfo.lincormaxbit)

           mask_S = (img_S lt (bias - double(bitinfo.lincormax)))* $
                    2^(bitinfo.lincormaxbit)

           mask = mc_combflagstack([[[mask_P]],[[mask_S]]], $
                                   bitinfo.lincormaxbit+1,CANCEL=cancel)
           
           bitmask[*,*,i] = rotate(mask,rot)
           
        endif 

;  Create real image
        
        img = img_P-img_S

;  Correct for amplifier offsets

        if doampcor then begin

           img = mc_ishellampcor(img,SILENT=1,CANCEL=cancel)
           if cancel then return
                      
        endif
        
;  Determine the correction for the image

        if keyword_set(LINCORR) then begin

;           print, lc_coeffs[1214,96,2:*]
           cor = mc_imgpoly(img,reform(lc_coeffs[*,*,2:*]),CANCEL=cancel)
           if cancel then return
           
;  Check for pixels above Saturation and set correction to unity.

           z = where(bitmask[*,*,i] eq 2^bitinfo.lincormaxbit,cnt)
           if cnt ne 0 then cor[z] = 1.0

;  Set black pixel corrections to unity as well.

           cor[0:3,*] = 1.0
           cor[2044:2047,*] = 1.0
           cor[*,0:3] = 1.0
           cor[*,2044:2047] = 1.0
           
;  Apply the corrections
           
           img = temporary(img)/cor
           
;  Delete unecessary files

           delvarx,cor,cor,img_S, img_P

        endif

;  Create the actual image.

;  Covert image back to total DN for error propagation

        img = temporary(img)*DIVISOR

        if dovar then begin

           variance = abs(img)*crtn/NDRS/(COADDS^2)/(ITIME^2)/gain + rdvar
        
        endif        

        img = temporary(img)/DIVISOR/ITIME

;  Rotate and store data and variance if requested

        data[*,*,i] = rotate(img,rot)
        if dovar then var[*,*,i] = rotate(variance,rot) 
        
        delvarx,img,variance

;  Store header information
        
        copy_struct_inx,getishellhdrinfo(hdr,KEYWORDS=keywords),hdrinfo, $
                        index_to=i
        
     endfor
     
  endif
  
end







