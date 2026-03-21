;+
; NAME:
;     mc_rectifyorder
;
; PURPOSE:
;     To rectify a spectral order.
;
; CALLING SEQUENCE:
;     result = mc_rectifyorder(img,indices,[var],[bdpxmk],FORWARDS=forwards,$
;                              BACKWARDS=backwards,CANCEL=cancel)
;
; INPUTS:
;     img     - A [ncols,nrows] image.
;     indices - 
;
; OPTIONAL INPUTS:
;     IVAR    - a 2D image (ncols,nrows) variance image to be rectified.
;     IBDPXMK - a 2D image (ncols,nrows) bad pixel mask to be rectified.
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;     A rectified order.
;
; OPTIONAL OUTPUTS:
;     OVAR    - a 2D image (ncols,nrows) variance image to be rectified.
;     OBDPXMK - a 2D image (ncols,nrows) bad pixel mask to be rectified.
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     None
;
; DEPENDENCIES:
;     Requires the Spextool package (and it dependencies)
;
; PROCEDURE:
;     If the /FORWARDS 
;
; EXAMPLES:
;     Later
;
; MODIFICATION HISTORY:
;     2017-03-06 - Written by M. Cushing, University of Toledo
;     2018-07-20 - Added a catch to look for bad pixel masks that have
;                  zeros in an entire column.  Needed when using
;                  wavecals from previous runs (probably due to a
;                  slight shift in wavelength).
;                               
;-
function mc_rectifyorder,img,indices,method,VARIMG=varimg,BPMASK=bpmask, $
                         BSMASK=bsmask,CANCEL=cancel
  
  cancel = 0

  s = size(img,/DIMEN)
  ncols = s[0]
  nrows = s[1]
  
;  Parse the indices array
  
  ix = reform(indices[1:*,1:*,0])
  iy = reform(indices[1:*,1:*,1])
  wgrid = reform(indices[1:*,0,0])
  sgrid = reform(indices[0,1:*,0])

;  Now do the interpolation
  
  if method eq 0 then begin

     rectimg = interpolate(img,ix,iy,MISSING=!values.f_nan)

     if keyword_set(VARIMG) then begin

        ovarimg = interpolate(varimg,ix,iy,MISSING=!values.f_nan)

     endif

     if keyword_set(BPMASK) then begin

        obpmask = interpolate(bpmask,ix,iy,MISSING=1)
        
     endif
     
     if keyword_set(BSMASK) then begin
        
        obsmask = interpolate(bsmask,ix,iy,MISSING=0)

     endif
             
  endif
    
  if method eq 1 then begin

     nw = n_elements(wgrid)-1
     ns = n_elements(sgrid)-1
     
;  Now set up output images
     
     rectimg = dblarr(nw,ns,/NOZERO)
     
     if keyword_set(VARIMG) then ovarimg = fltarr(nw,ns,/NOZERO)
     if keyword_set(BPMASK) then obpmask = make_array(nw,ns,/BYTE,VALUE=1)
     if keyword_set(BSMASK) then obsmask = make_array(nw,ns,/BYTE,VALUE=0)

     wgrid = ((wgrid+shift(wgrid,-1))/2D)[0:(nw-1)]
     sgrid = ((sgrid+shift(sgrid,-1))/2D)[0:(ns-1)]
     
;  Now loop over each slit polygon
     
     for j = 0,nw-1 do begin
        
;  Concatinate the corners of each new pixel in the slit polygon to do
;  multiple polygons at once.  Loop over each pixel for now, be
;  creative later.
        
        px = 0
        py = 0
        
        for k = 0,ns-1 do begin
           
           px = [px,reform(ix[j,k:k+1]),reverse(reform(ix[j+1,k:k+1]))]
           py = [py,reform(iy[j,k:k+1]),reverse(reform(iy[j+1,k:k+1]))]
           
        endfor
        
        px = px[1:*]+0.5
        py = py[1:*]+0.5
        
        ri = lindgen(ns+1)*4
        
;  (the +0.5 above is because J.D. should be killed for indexing
;  pixels at the lower left hand corner instead of the middle)
        
        inds = polyfillaa(px,py,ncols,nrows,AREAS=areas,POLY_INDICES=ri)
     
;  Reconstruct the spatial profile at this wavelength
     
        for k =0,ns-1 do begin
           
           idx = inds[ri[k]:ri[k+1]-1]
           area = areas[ri[k]:ri[k+1]-1]
           
           rectimg[j,k] = total(img[idx]*area)
           
           if keyword_set(VARIMG) then ovarimg[j,k] = total(varimg[idx]*area)
           if keyword_set(BPMASK) then obpmask[j,k] = product(bpmask[idx],/PRE)
           if keyword_set(BSMASK) then $
              obsmask[j,k] = (total(bsmask[idx]) gt 0) ? 1 : 0  
           
        endfor
        
     endfor

  endif

;  Check for columns of pixels that are completely zero.  This occurs
;  if you are using a arc/flat from a different observationa resulting
;  in potentially a slight shift in what pixels are ok.  
  
  if keyword_set(BPMASK) then begin

     mask = total(obpmask,2)

     minidx = min(where(mask ne 0),MAX=maxidx)
     
     rectimg = rectimg[minidx:maxidx,*]
     obpmask = obpmask[minidx:maxidx,*]
     wgrid = wgrid[minidx:maxidx]

     if keyword_set(VARIMG) then ovarimg = ovarimg[minidx:maxidx,*]
     if keyword_set(BSMASK) then obsmask = obsmask[minidx:maxidx,*]
     
  endif
  
  out = {img:rectimg,wgrid:wgrid,sgrid:sgrid}

  if keyword_set(VARIMG) then out = create_struct(out,'var',ovarimg)
  if keyword_set(BPMASK) then out = create_struct(out,'bpm',obpmask)
  if keyword_set(BSMASK) then out = create_struct(out,'bsm',obsmask)
  
  return, out
  
end
