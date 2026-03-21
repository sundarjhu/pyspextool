;+
; NAME:
;     mc_mkspatprof
;
; PURPOSE:
;     To construct average spatial profiles.
;
; CALLING SEQUENCE:
;     result = mc_mkspatprof(img,omask,wavecal,spatcal,edgecoeffs,ds,$
;                            [wtrans],[ttrans],[atmosthresh],$
;                            SPATCOEFFS=spatcoeffs,$
;                            MEDSPATCOEFFS=medspatcoeffs,UPDATE=update,$
;                            WIDGET_ID=widget_id,CANCEL=cancel)
;
; INPUTS:
;     img         - a 2D image.
;     omask       - A 2D image of the same size as img that gives the
;                   order number of each pixel.
;     wavecal     - A 2D image of the same size as img that gives the
;                   wavelength of each pixel.
;     spatcal     - A 2D image of the same size as img that gives the
;                   spatial coordinates of each pixel.
;     edgecoeffs  - Array [degree+1,2,norders] of polynomial coefficients 
;                   which define the edges of the orders.  array[*,0,0]
;                   are the coefficients of the bottom edge of the
;                   first order and array[*,1,0] are the coefficients 
;                   of the top edge of the first order.
;     ds          - The spatial spacing of the resampling slit in
;                   arcsec, typically given by slith_arc/slith_pix.
;     ybuffer     - The number of pixels to ignore near the top and
;                   bottom of the slit.  
;
; OPTIONAL INPUTS:
;     wtrans      - An 1D array of wavelengths for the atmospheric transmission.
;     ttrans      - A 1D array of the the atmospheric transmission.
;     atmosthresh - The transmission (0-1) below which data is ignored.
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS: 
;     Returns a structure with norders fields containing the median
;     spatial profiles.  Each field is a 2D array where array[*,0] is 
;     contains the x values in arcseconds and array[*,1] are the
;     intensity values.  .
;
; OPTIONAL OUTPUTS:
;     SPATCOEFFS    - A structure with 2*norders elements containing the
;                     the information necessary to construct the spatial
;                     map.  The first field of the two fields is a 1D
;                     array  [nspat] containing the values of the
;                     spatial  coordinate.  The second field contains
;                     an  array [3, nspat] of polynomial cofficients.
;                     The coefficients are a function of wavelength
;                     (whether it be actually units or pixels).
;     MEDSPATCOEFFS - Same as SPATCOEFFS except that the polynomial
;                     coefficients give the median spatial profile
;                     instead of a wavelength dependent profile.
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     
;
; DEPENDENCIES:
;     mc_cpar (Spextool)
;     poly1d (Spextool)
;     linterp (astron)
;
; PROCEDURE:
;     Each order is resampled onto a uniform grid.  The median
;     background is subtracted on a column by column basis.  The
;     median spatial profile is then created.  If the user passes
;     wtrans, ttrans, atmosthresh, then pixels that have atmospheric
;     transmission below atmosthresh are ignored.  The median spatial
;     profile is then used to normalize the resampling image.
;     2D polynomial coefficients are then derived on a row by row basis.     
;
; EXAMPLES:
;     NA     
;
; MODIFICATION HISTORY:
;     2009-12-03 - Written by M. Cushing, NASA JPL
;     2017-03-07 - Added the edgecoeffs parameters.
;     2017-08-14 - Heavily modified to incorporate the new iSHELL methods.
;-
function mc_mkspatprof,rectimgs,orders,wtrans,ttrans,UPDATE=update,CANCEL=cancel
  
  debug = 0
  cancel = 0

;  if n_params() lt 8 then begin
;
;     print, 'Syntax - mc_mkspatprof2d(img,omask,wavecal,spatcal,ds,ybuffer,$'
;     print, '                         [wtrans],[ttrans],[atmosthresh],$'
;     print, '                         SPATCOEFFS=spatcoeffs,$'
;     print, '                         MEDSPATCOEFFS=medspatcoeffs,$'
;     print, '                         UPDATE=update,WIDGET_ID=widget_id,$'
;     print, '                         CANCEL=cancel)'
;     cancel = 1
;     return, -1
;
;  endif

;  cancel = mc_cpar('mc_mkspatprof2d',img, 1,'Wcinfo',[2,3,4,5],2)
;  if cancel then return, -1
;  cancel = mc_cpar('mc_mkspatprof2d',omask, 2,'Omask',[2,3,4,5],2)
;  if cancel then return, -1
;  cancel = mc_cpar('mc_mkspatprof2d',wavecal, 3,'Wavecal',[2,3,4,5],2)
;  if cancel then return, -1
;  cancel = mc_cpar('mc_mkspatprof2d',spatcal, 4,'Spatcal',[2,3,4,5],2)
;  if cancel then return, -1
;  cancel = mc_cpar('mc_mkspatprof2d',ds, 5,'ds',[2,3,4,5],0)
;  if cancel then return, -1
;  cancel = mc_cpar('mc_mkspatprof2d',ybuffer, 6,'ybuffer',[2,3,4,5],0)
;  if cancel then return, -1
;
;  if n_elements(atmosthresh) ne 0 then begin
;          
;     cancel = mc_cpar('mc_mkspatprof2d',wtrans,7,'WTrans',[2,3,4,5],1)
;     if cancel then return,-1
;     cancel = mc_cpar('mc_mkspatprof2d',ttrans,8,'TTrans',[2,3,4,5],1)
;     if cancel then return,-1
;     cancel = mc_cpar('mc_mkspatprof2d',atmosthresh,9,'Atmosthresh',[2,3,4,5],0)
;     if cancel then return,-1
;
;  endif

  if debug then begin
     
     window, /FREE
     wid = !d.window
     re = ' '
     
  endif

  norders = n_elements(orders)
  
;  Loop over each order

  for i = 0,norders-1 do begin

     rectorder = (rectimgs.(i))[1:*,1:*]
     xgrid     = reform((rectimgs.(i))[1:*,0])
     ygrid     = reform((rectimgs.(i))[0,1:*])
     
     s = size(rectorder,/DIMEN)
     nx = s[0]
     ny = s[1]
     
;  Subtract background

     bg = median(rectorder,DIMEN=2,/EVEN)
     rectorder = temporary(rectorder)-rebin(bg,nx,ny)

;  Compute the profile

     prof = fltarr(ny,/NOZERO)
     if n_elements(wtrans) ne 0 then begin

;  Weight by atmosphere is you can
        
        linterp,wtrans,ttrans,xgrid,rtrans,MISSING=1
        rtrans = (rtrans > 0.01)^2
        datavar = (1/rtrans)/total(1/rtrans)
        
     endif

;  Do the combining

     for j = 0,ny-1 do begin

        mc_meancomb,rectorder[*,j],mean,DATAVAR=datavar,ROBUST=4,/NAN, $
                    /SILENT,CANCEL=cancel
        if cancel then return,-1
        prof[j] = mean
        
     endfor

;  Normalize by total absolute flux
     
     prof = prof/total(abs(prof),/NAN)

;  Store the results in a structure

     key = 'Order'+string(orders[i],FORMAT='(I3.3)')
     sprofiles = (i eq 0) ? create_struct(key,[[ygrid],[prof]]):$
                 create_struct(sprofiles,key,[[ygrid],[prof]])

  endfor
  
  if cancel then return, -1 else return, sprofiles

end
