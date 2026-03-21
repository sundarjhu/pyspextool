;+
; NAME:
;     mc_readmodefile
;
; PURPOSE:
;     Reads a mode calibration file for Spextool.
;
; CALLING SEQUENCE:
;     result = mc_readmodefile(filename,CANCEL=cancel)
;
; INPUTS:
;     filename - The full path of a Spextool _flatinfo file
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;     result - a structure where the tag if each field is given below:
;
;              modename        - A string giving the mode name.
;              rotation        - The IDL rotation direction 
;              orders          - An array of the order numbers.
;              ps              - Plate scale in arcsec pix-1
;              slith_pix       - The slit height in pixels. 
;              slith_pix_range - The slit height range in pixels. 
;              slith_arc       - The slit height in arcseconds.
;              rppix           - The resolving power times the slit width in
;                                pixels for any given slit in this mode.  Used
;                                to scale to other slit widths.
;              step            - The step size in the dispersion direction 
;                                (see findorders) 
;              flatfrac        - The fraction value used to find the edges of
;                                the orders (see findorders.pro)
;              comwin          - The center-of-mass searching window.
;              edgedeg         - The polynomial degree of the edges of
;                                the  orders.
;              norm_nxg        - The number of grid squares in the x
;                                direction used
;                                to removed the scattered light
;              norm_nyg        - The number of grid squares in the y
;                                direction used
;                                to removed the scattered light
;              oversamp        - The over sampling factor when it straightens
;                                each order
;              ybuffer         - The number of pixels to move inside the edges
;                                of the orders since the edges are not
;                                inifitely sharp
;              fixed           - If 'Yes' then the guesspos are row
;                                numbers  of the edge of the order.
;                                If 'No',  then they are guess positions.
;              guesspos        - An (2,norders) array of positions of
;                                the  orders on the array.
;              findrange       - An (2,norders) array of xranges to search for
;                                the orders
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
;     None
;
; PROCEDURE:
;     Easy
;
; EXAMPLE:
;    
; MODIFICATION HISTORY:
;     2000-05-10 - Written by M. Cushing, Institute for Astronomy, UH
;     2002-09-01 - Added slith_pix_range parameter
;     2003-01-23 - Removed SG stuff and added fiterpolate stuff,
;                  yubuffer
;     2007-04-02 - Added findrange output variable.
;     2007-07-13 - Added flatfrac input 
;     2007-07-14 - Added step input
;     2007-07-25 - Removed start and stop inputs.
;     2007-07-25 - Added comwin input
;     2008-02-22 - Added rppix input.
;     2009-12-16 - Added linedeg,linereg,ystep,ysum,wxdeg,wydeg,sxdeg,sydeg
;     2010-08-08 - Changed to a function, changed output to a
;                  structure, added type, homeorder, dispdeg, and
;                  ordrdeg output, and renamed to mc_readmodfile.
;     2017-10-18 - Rewritten so that the keywords can be in any order.
;-
function mc_readmodefile,filename,CANCEL=cancel

  cancel = 0
  
;  Check parameters
  
  if n_params() lt 1 then begin
     
     print, 'Syntax - result = mc_readmodefile(filename,CANCEL=cancel)'
     cancel = 1
     return, -1
     
  endif
  
  cancel = mc_cpar('mc_readmodefile',filename,1,'Filename',7,0)
  if cancel then return,-1
  
  readcol,filename,key,val,COMMENT='#',DELIMITER='=',FORMAT='A,A',/SILENT
  key = strtrim(key,2)
  
;  MODENAME

  z = where(key eq 'MODENAME',cnt)
  if cnt ne 0 then begin

     str = {modename:strjoin(strtrim(val[z],2))}

  endif else begin
     
     print, 'mc_readmodefile:  MODENAME not found.'
     cancel = 1
     return, -1

  endelse

;  ROTATION

  z = where(key eq 'ROTATION',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'rotation',total(fix(val[z]),/PRESERVE))   

  endif else begin
     
     print, 'mc_readmodefile:  ROTATION not found.'
     cancel = 1
     return, -1

  endelse  

;  ORDERS

  z = where(key eq 'ORDERS',cnt)
  if cnt ne 0 then begin

     orders = fix(mc_fsextract(val[z[0]],/INDEX,CANCEL=cancel))
     if cancel then return,-1     
     str = create_struct(str,'orders',orders)     

  endif else begin
     
     print, 'mc_readmodefile:  ORDERS not found.'
     cancel = 1
     return, -1

  endelse  

;  PLATE SCALE

  z = where(key eq 'PLATE SCALE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'ps',total(float(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  PLATE SCALE not found.'
     cancel = 1
     return, -1

  endelse

;  SLITH_PIX

  z = where(key eq 'SLITH_PIX',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'slith_pix',total(float(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  SLITH_PIX not found.'
     cancel = 1
     return, -1

  endelse

;  SLITH_PIX RANGE

  z = where(key eq 'SLITH_PIX_RANGE',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'slith_pix_range',fix(strsplit(val[z],' ')))     
     
  endif else begin
     
     print, 'mc_readmodefile:  SLITH_PIX_RANGE not found.'
     cancel = 1
     return, -1

  endelse        

;  SLITH_ARC

  z = where(key eq 'SLITH_ARC',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'slith_arc',total(float(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  SLITH_ATC not found.'
     cancel = 1
     return, -1

  endelse

;  RPPIX

  z = where(key eq 'RPPIX',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'rppix',total(float(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  RPPIX not found.'
     cancel = 1
     return, -1

  endelse

;  STEP

  z = where(key eq 'STEP',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'step',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  STEP not found.'
     cancel = 1
     return, -1

  endelse

;  FLATFRAC

  z = where(key eq 'FLATFRAC',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'flatfrac',total(float(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  FLATFRAC not found.'
     cancel = 1
     return, -1

  endelse

;  COMWIN 

  z = where(key eq 'COMWIN',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'comwin',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  COMWIN not found.'
     cancel = 1
     return, -1

  endelse

;  EDGEDEG

  z = where(key eq 'EDGEDEG',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'edgedeg',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  EDGEDEG not found.'
     cancel = 1
     return, -1

  endelse

;  NORM_NXG

  z = where(key eq 'NORM_NXG',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'norm_nxg',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  NORM_NXG not found.'
     cancel = 1
     return, -1

  endelse  

;  NORM_NYG

  z = where(key eq 'NORM_NYG',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'norm_nyg',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  NORM_NYG not found.'
     cancel = 1
     return, -1

  endelse

;  OVERSAMP

  z = where(key eq 'OVERSAMP',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'oversamp',total(float(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  OVERSAMP not found.'
     cancel = 1
     return, -1

  endelse    

;  NORM_YBUFFER

  z = where(key eq 'NORM_YBUFFER',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'norm_ybuffer',total(fix(val[z]),/PRESERVE))     

  endif else begin
     
     print, 'mc_readmodefile:  NORM_YBUFFER not found.'
     cancel = 1
     return, -1

  endelse

;  FIXED

  z = where(key eq 'FIXED',cnt)
  if cnt ne 0 then begin

     str = create_struct(str,'fixed',strtrim(val[z],2))

  endif else begin
     
     print, 'mc_readmodefile:  FIXED not found.'
     cancel = 1
     return, -1

  endelse      

;  GUESSPOS

  norders = n_elements(str.orders)
  guesspos = intarr(2,norders,/NOZERO)
  for i = 0,norders-1 do begin

     keyword = 'GUESSPOS_'+strtrim(orders[i],2)
     z = where(key eq keyword,cnt)
     if cnt ne 0 then begin

        guesspos[*,i] = fix( strsplit(val[z],' ',/EXTRACT))       
        
     endif else begin
        
        print, 'mc_readmodefile:  '+keyword+' not found.'
        cancel = 1
        return, -1
        
     endelse      
        
  endfor
  
  str = create_struct(str,'guesspos',guesspos)

;  ORDERRANGE

  range = intarr(2,norders,/NOZERO)
  for i = 0,norders-1 do begin

     keyword = 'ORDERRANGE_'+strtrim(orders[i],2)
     z = where(key eq keyword,cnt)
     if cnt ne 0 then begin

        range[*,i] = fix( strsplit(val[z],' ',/EXTRACT))       
        
     endif else begin
        
        print, 'mc_readmodefile:  '+keyword+' not found.'
        cancel = 1
        return, -1
        
     endelse      
        
  endfor
  
  str = create_struct(str,'findrange',range)

  return, str

end
