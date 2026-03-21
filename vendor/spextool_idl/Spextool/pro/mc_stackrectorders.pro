;+
; NAME:
;     mc_stackrectorders
;
; PURPOSE:
;     To stack rectified spectral orders into an image.
;
; CALLING SEQUENCE:
;     result = mc_stackrectorders(imgs,orders,slith_arc,offset,VARS=vars, $
;                                 BPMASKS=bpmasks,BSMASKS=bsmasks,
;                                 OFFSETS=offsets,CANCEL=cancel)
;
; INPUTS:
;     imgs      - A structure with [norders] tags.  Each tag contains
;                 a 2D array with the rectified image, wavelength
;                 array, and sky angles array.  The array can be
;                 visualized as follows:
;
;                  a  i i i i i i i i i i i i
;                  a  i i i i i i i i i i i i
;                  a  i i i i i i i i i i i i
;                  a  i i i i i i i i i i i i
;                  a  i i i i i i i i i i i i
;                 NaN w w w w w w w w w w w w
;
;                 where a is the angular position along the slit, w,
;                 is a wavelength, and i is an image value.  The values
;                 for the first order can be accessed as follow:
;
;                 img  = (img.(0))[1:*,1:*]
;                 wave = (img.(0))[1:*,0]
;                 ang  = (img.(0))[0,1:*]
;
;     orders    - Integer array of the order numbers
;     slith_arc - Float scalar giving the slit height in arcseconds
;     offset    - Number of pixels between orders in the output image
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     VARS    - A structure identical in form to imgs but for the
;               variance images.
;     BPMASKS - A structure identical in form to imgs but for a bad
;               pixel mask.
;     BSMASKS - A structure identical in form to imgs but for a bit-set
;               mask.
;     CANCEL  - Set on return if there is a problem.
;
; OUTPUTS:
;     result - {img:img,omask:omask,wavecal:wavecal,spatcal:spatcal, $
;                edgecoeffs:edgecoeffs,xranges:xranges}
;
;               img        = 2D image with the orders stacked on top of each
;                            other
;               omask      = 2D image where each pixel is set to its order
;                            number
;               wavecal    = 2D image where each pixel is set to its
;                            wavelength
;               spatcal    = 2D image where each pixel is set to its
;                            anglular position on the sky
;               edgecoeffs = Array [degree+1,2,norders] of polynomial
;                            coefficients which define the edges of
;                            the orders.  array[*,0,0] are the
;                            coefficients of the bottom edge of the
;                            first order and array[*,1,0] are the
;                            coefficients of the top edge of the first order.
;               xranges    = An array [2,norders] of column numbers where the
;                            orders are completely on the array
;
; OPTIONAL OUTPUTS:
;     if VARS is passed, an additional tag "var" is added to the
;     structure which holds a 2D variance image.  If BPMASKS is
;     passed, an additional tag "bpm" is added to the structure which
;     holds the bad pixel mask, and if BSMASKS is passed, an
;     additional tag "bsm" is added to the structure which holds the
;     bit-set mask.
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
;     Parses the imgs structure and then creates 2D iamges.
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-08 - Written by M. Cushing, University of Toledo
;-
function mc_stackrectorders,imgs,orders,slith_arc,offset,VARS=vars, $
                            BPMASKS=bpmasks,BSMASKS=bsmasks,OFFSETS=offsets, $
                            CANCEL=cancel
  
  cancel = 0

;  Check parameters

  if n_params() lt 4 then begin

     print, 'Syntax - result = mc_stackrectorders(imgs,orders,slith_arc,$'
     print, '                  offset,VARS=vars,BPMASKS=bpmasks,$'
     print, '                  BSMASKS=bsmasks,OFFSETS=offsets,CANCEL=cancel'
     cancel = 1
     return,-1
     
  endif
  
  cancel = mc_cpar('mc_stackrectorders',imgs,1,'Image',8,1)
  if cancel then return,-1
  cancel = mc_cpar('mc_stackrectorders',orders,2,'Orders',[2,3],[0,1])
  if cancel then return,-1
  cancel = mc_cpar('mc_stackrectorders',slith_arc,3,'Slith_arc',[2,3,4,5],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_stackrectorders',offset,4,'Offset',[2,3],0)
  if cancel then return,-1      
  
; Get norders and the sizes of the images
  
  norders = n_elements(orders)
  s = intarr(2,norders,/NOZERO)
  for i = 0,norders-1 do s[*,i] = size((imgs.(i))[1:*,1:*],/DIMEN)
  
;  Get the spatial grid
  
  sgrid = reform(imgs.(0)[0,1:*])

;  Get ready for the edgecoeff creation
  
  edgecoeffs = fltarr(2,2,norders)
  x = findgen(s[1,0])
  bottop = interpol(x,sgrid,[0,slith_arc])
     
;  Now set up the stack arrays

  nx = max(s[0,*])
  ny = s[1,0]*norders+offset*(norders-1)+2*offset
  
  img     = make_array(nx,ny,/DOUBLE,VALUE=!values.f_nan)
  wavecal = make_array(nx,ny,/DOUBLE,VALUE=!values.f_nan)
  spatcal = make_array(nx,ny,/DOUBLE,VALUE=!values.f_nan)
  omask   = make_array(nx,ny,/INTEGER,VALUE=0)
  offsets = intarr(norders)

  if keyword_set(VARS) then var = make_array(nx,ny,/DOUBLE,VALUE=!values.f_nan)
  if keyword_set(BPMASKS) then bpmask = make_array(nx,ny,/BYTE,VALUE=0)
  if keyword_set(BSMASKS) then bsmask = make_array(nx,ny,/BYTE,VALUE=0)
  
  for i = 0,norders-1 do begin

     ystart = (offset+s[1,i]*i+offset*i)
     offsets[i] = ystart
     
     wgrid = reform(imgs.(i)[1:*,0])
     
     img[0,ystart]     = (imgs.(i))[1:*,1:*]
     wavecal[0,ystart] = rebin(wgrid,s[0,i],s[1,i])
     spatcal[0,ystart] = rebin(reform(sgrid,1,s[1,i]),s[0,i],s[1,i])
     omask[0,ystart]   = make_array(s[0,i],s[1,i],/INTEGER,VALUE=orders[i])

     edgecoeffs[0,0,i] = bottop[0]+ystart
     edgecoeffs[0,1,i] = bottop[1]+ystart

     if keyword_set(VARS) then var[0,ystart] = (vars.(i))[1:*,1:*]
     if keyword_set(BPMASKS) then bpmask[0,ystart] =byte((bpmasks.(i))[1:*,1:*])
     if keyword_set(BSMASKS) then bsmask[0,ystart] =byte((bsmasks.(i))[1:*,1:*])
     
  endfor

;  Create the xranges

  xranges = make_array(2,norders,/INTEGER,VALUE=0)
  xranges[1,*] = s[0,*]-1

;  Creat the output

  result = {img:img,omask:omask,wavecal:wavecal,spatcal:spatcal, $
            edgecoeffs:edgecoeffs,xranges:xranges}

  if keyword_set(VARS) then result = create_struct(result,'var',var)
  if keyword_set(BPMASKS) then result = create_struct(result,'bpm',bpmask)
  if keyword_set(BSMASKS) then result = create_struct(result,'bsm',bsmask)
  
  return, result
  
end
