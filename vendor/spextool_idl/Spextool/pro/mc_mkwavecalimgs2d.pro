;+
; NAME:
;     mc_mkwavecalimgs2d
;
; PURPOSE:
;     Create two arrays where pixels are set to their wavelength and
;     spatial coordinate values.
;
; CALLING SEQUENCE:
;     mc_mkwavecalimgs2d,omask,orders,indices,wavecal,spatcal,UPDATE=update,$
;                        WIDGET_ID=widget_id,CANCEL=cancel     
;
; INPUTS:
;     omask   - An [ncols,nrows] array where each pixel value is set
;               to the order number.  Interorder pixels are set to zero.
;     orders  - An [norders] array giving the order numbers.
;     indices - A structure with norders tags.  Each tag consists of
;               an [nxgrid,nygrid,2] array.  The [*,*,0] array gives the
;               x coordinates of lines of constant wavelength (see
;               wgrids parameter) and spatial coorindate (see sgrid
;               parameter) while the [*,*,1] gives the y coordinates.  
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     UPDATE     - If set, the progress of the code will be displayed
;                  either in the terminal or the Fanning showprogress widget
;                  (see keyword WIDGET_ID).
;     WIDGET_ID  - If given, a cancel button is added to the Fanning
;                  showprogress routine.  The widget blocks the
;                  WIDGET_ID, and checks for for a user cancel
;                  command.
;     CANCEL     - Set on return in there is a problem.
;
; OUTPUTS:
;     wavecal - A 2D image wherein each pixel within an order is set
;               to its wavelength.  Interorder pixels are set to NaN.
;     spatcal - A 2D image wherein each pixel within an order is set
;               to its spatial coordinate value.   Interorder pixels
;               are set to NaN.
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
;     Spextool package (and its dependencies)
;
; PROCEDURE:
;     
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-08-13:  Written by M. Cushing, University of Toledo
;     2019-04-31: Renamed mc_mkwavecalimages2d.
;-
pro mc_mkwavecalimgs2d,omask,orders,indices,wavecal,spatcal,CANCEL=cancel

  cancel = 0

;  Get sizes of arrays
  
  s = size(omask,/DIMEN)
  ncols = s[0]
  nrows = s[1]

  xx = rebin(indgen(ncols),ncols,nrows)
  yy = rebin(reform(indgen(nrows),1,nrows),ncols,nrows)  

  norders = n_elements(orders)

;  Create blank wavecal and spatcal images
  
  wavecal = make_array(ncols,nrows,/DOUBLE,VALUE=!values.f_nan)
  spatcal = make_array(ncols,nrows,/DOUBLE,VALUE=!values.f_nan)
  
;  Start the loop over the orders
  
  for i = 0,norders-1 do begin

;  Set up the temoporary image to make transfering the results to the
;  wavecal and spatcal arrays simplier (translation:  haven't
;  figured out how to do it slickly with indices).
     
     tmpimg = dblarr(ncols,nrows,/NOZERO)

;  Pull the indices and wgrid for this order
     
     ix = reform(indices.(i)[1:*,1:*,0])
     iy = reform(indices.(i)[1:*,1:*,1])
     wgrid = reform(indices.(i)[1:*,0,0])
     sgrid = reform(indices.(i)[0,1:*,0])
     
     s = size(ix,/DIMEN)

;  Generate wavelength and spatial coordinate arrays corresponding to
;  the indices array.
     
     iw = rebin(wgrid,s[0],s[1])
     is= rebin(reform(sgrid,1,s[1]),s[0],s[1])

;  Figure out the column and row range over which you are going to do
;  the interpolation.
     
     zordr = where(omask eq orders[i])
     xmax = max(xx[zordr],MIN=xmin)
     ymax = max(yy[zordr],MIN=ymin)
     
;  Generate a uniform grid in x and y
     
     xgrid = findgen(xmax-xmin+1)+xmin
     ygrid = findgen(ymax-ymin+1)+ymin
     
;  Perform the interpolation
     
     w = griddata(ix,iy,iw,/POLYNOMIAL,DEG=3,XOUT=xgrid,YOUT=ygrid,$
                  /GRID,MISSING=!values.f_nan)
     
     s = griddata(ix,iy,is,/POLYNOMIAL,DEG=3,XOUT=xgrid,YOUT=ygrid,$
                  /GRID,MISSING=!values.f_nan)

;  Place these arrays in the tmpimg array and then copy the correct
;  pixels in the wavecal and spatcal arrays.
     
     tmpimg[xmin:xmax,ymin:ymax] = w
     wavecal[zordr] = tmpimg[zordr]
     tmpimg[xmin:xmax,ymin:ymax] = s
     spatcal[zordr] = tmpimg[zordr]

  endfor

end
