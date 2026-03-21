;+
; NAME:
;     mc_readflatinfo
;
; PURPOSE:
;     Read a Spextool flatinfo file.
;
; CATEGORY:
;     Spectroscopy
;
; CALLING SEQUENCE:
;     result = mc_readflatinfo(filename,CANCEL=cancel)
;
; INPUTS:
;     filename - The full path of a Spextool calibration file
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
;              modename    - A string giving the mode name.
;              rotation    - The IDL rotation direction 
;              norders     - The number of orders in the image.
;              orders      - An array of the order numbers.
;              ps          - Plate scale in arcsec pix-1
;              slith_pix   - The slit height in pixels. 
;              slith_range - The slit height range in pixels. 
;              slith_arc   - The slit height in arcseconds.
;              rppix       - The resolving power times the slit width in
;                            pixels for any given slit in this mode.  Used
;                            to scale to other slit widths.
;              step        - The step size in the dispersion direction 
;                            (see findorders) 
;              flatfrac    - The fraction value used to find the edges of
;                            the orders (see findorders.pro)
;              comwin      - The center-of-mass searching window.
;              edgedeg     - The polynomial degree of the edges of
;                            the  orders.
;              norm_nxg    - The number of grid squares in the x
;                            direction used
;                            to removed the scattered light
;              norm_nyg    - The number of grid squares in the y
;                            direction used
;                            to removed the scattered light
;              oversamp    - The over sampling factor when it straightens
;                            each order
;              ybuffer     - The number of pixels to move inside the edges
;                            of the orders since the edges are not
;                            inifitely sharp
;              fixed       - If 'Yes' then the guesspos are row
;                            numbers  of the edge of the order.
;                            If 'No',  then they are guess positions.
;              ycorordr    - The order number with which to do the
;                            Y cross correlation
;              guesspos    - An (2,norders) array of positions of
;                            the  orders on the array.
;              findrange   - An (2,norders) array of xranges to search for
;                            the orders
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
;     2017-03-31 - Removed homeorder as that belongs in the linecal
;                  file.
;     2017-05-11 - Removed a bunch of wavecal stuff.
;     2017-12-30 - Modified to read the new FITS format.
;-
function mc_readflatinfo,filename,CANCEL=cancel

  cancel = 0
  
;  Check parameters
  
  if n_params() lt 1 then begin
     
     print, 'Syntax - result = mc_readmodefile(filename,CANCEL=cancel)'
     cancel = 1
     return, -1
     
  endif
  
  cancel = mc_cpar('mc_readmodefile',filename,1,'Filename',7,0)
  if cancel then return,-1

  omask = readfits(filename,hdr,/SILENT)
  
  modename = fxpar(hdr,'MODENAME')
  str = {modename:modename}

  val = fix(fxpar(hdr,'ROTATION'))
  str = create_struct(str,'rotation',val)

  val = fxpar(hdr,'SLTH_ARC')
  str = create_struct(str,'slith_arc',val)

  val = fxpar(hdr,'SLTH_PIX')  
  str = create_struct(str,'slith_pix',val)

  val = fix(strsplit(fxpar(hdr,'SLTH_RNG'),',',/EXTRACT))
  str = create_struct(str,'slith_range',val)    

  orders = long( strsplit( fxpar(hdr,'ORDERS'), ',', /EXTRACT) )
  str = create_struct(str,'orders',orders)      
  norders = n_elements(orders)
  
  val = fxpar(hdr,'RPPIX')  
  str = create_struct(str,'rpppix',val)

  val = fxpar(hdr,'PLTSCALE')  
  str = create_struct(str,'ps',val)  

  val = fix(fxpar(hdr,'FIXED'))
  str = create_struct(str,'fixed',val)

  if ~val then begin
  
     val = fix(fxpar(hdr,'STEP'))
     str = create_struct(str,'step',val)
     
     val = fxpar(hdr,'FLATFRAC')  
     str = create_struct(str,'flatfrac',val)
     
     val = fix(fxpar(hdr,'COMWIN'))
     str = create_struct(str,'comwin',val)

  endif
     
  val = fix(fxpar(hdr,'EDGEDEG'))
  str = create_struct(str,'edgedeg',val)
          
  val = fix(fxpar(hdr,'NORM_NXG'))
  str = create_struct(str,'norm_nxg',val)

  val = fix(fxpar(hdr,'NORM_NYG'))  
  str = create_struct(str,'norm_nyg',val)

  val = fxpar(hdr,'OVERSAMP')  
  str = create_struct(str,'oversamp',val)

  val = fix(fxpar(hdr,'YBUFFER'))
  str = create_struct(str,'ybuffer',val)


  val = fix(fxpar(hdr,'YCORORDR'))
  str = create_struct(str,'ycororder',val)                    

;  Now get the edgecoefficients and xranges

  xranges    = intarr(2,norders)   
  edgecoeffs = dblarr(str.edgedeg+1,2,norders)
  guesspos   = intarr(2,norders)

  for i = 0,norders-1 do begin
     
     name_T = 'OR'+string(orders[i],FORMAT='(i3.3)')+'_T*'
     name_B = 'OR'+string(orders[i],FORMAT='(i3.3)')+'_B*'
     
     coeff_T = fxpar(hdr,name_T)
     coeff_B = fxpar(hdr,name_B)

     edgecoeffs[*,1,i] = coeff_T
     edgecoeffs[*,0,i] = coeff_B
     
     name         = 'OR'+string(orders[i],FORMAT='(i3.3)')+'_XR'
     xranges[*,i] = long( strsplit( fxpar(hdr,name), ',', /EXTRACT) )

     guesspos[0,i] = total(xranges[*,i])/2

     botedge = mc_poly1d(guesspos[0,i],edgecoeffs[*,0,i])
     topedge = mc_poly1d(guesspos[0,i],edgecoeffs[*,1,i])

     guesspos[1,i] = (botedge+topedge)/2.

     
  endfor

  str = create_struct(str,'xranges',xranges)
  str = create_struct(str,'edgecoeffs',edgecoeffs)
  str = create_struct(str,'guesspos',guesspos)
  str = create_struct(str,'omask',omask)                    

  
  return,str


end
