;+
; NAME:
;     mc_mkwavecalimgs1d
;
; PURPOSE:
;     To create a wavecal and spatcal file for (Spextool) 1D extraction 
;
; CALLING SEQUENCE:
;     mc_mkwavecalimgs1d,ncols,nrows,edgecoeffs,xranges,slith_arc,wavecal,$
;                        spatacal,CANCEL=cancel
;
; INPUTS:
;     ncols       - The number of columns in the image.
;     nrows       - The number of rows in the image.
;     edgecoeffs  - Array [degree+1,2,norders] of polynomial coefficients 
;                   which define the edges of the orders.  array[*,0,0]
;                   are the coefficients of the bottom edge of the
;                   first order and array[*,1,0] are the coefficients 
;                   of the top edge of the first order.
;     xranges     - An array [2,norders] of pixel positions where the
;                   orders are completely on the array
;     ybuffer     - Number of pixels to move inside of the edge of
;                   array since the edges aren't infinitely sharp
;     slith_arc   - The slit height in arcseconds.
;
; OPTIONAL INPUTS:
;     none
;
; KEYWORD PARAMETERS:
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;     wavecal - A 2D array where each pixel is set to its wavelength
;               (column in this case).
;     spatcal - A 2D array where each pixel is set to its angular
;               position on the sky
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
;     Loops over each order.  Using the edgecoeffs, and on a column by
;     column basis, it fills the wavecal and spatcal files.
;
; EXAMPLES:
;
;
; MODIFICATION HISTORY:
;     2019-04-31 - Written by M. Cushing, University of Toledo
;-
pro mc_mkwavecalimgs1d,ncols,nrows,edgecoeffs,xranges,wgrid,slith_arc, $
                       wavecal,spatcal,CANCEL=cancel
  
  cancel = 0
  
  if n_params() lt 6 then begin
     
     print, 'Syntax - mc_mkwavecalimgs1d,ncols,nrows,edgecoeffs,xranges,$'
     print, '                            slith_arc,wavecal,spatcal,$'
     print, '                            CANCEL=cancel'
     cancel = 1
     return

  endif

  cancel = mc_cpar('mc_mkwavecalimgs1d',ncols, 1,'Ncols',[2,3],0)
  if cancel then return
  cancel = mc_cpar('mc_mkwavecalimgs1d',nrows, 2,'Nrows',[2,3],0)
  if cancel then return
  cancel = mc_cpar('mc_mkwavecalimgs1d',edgecoeffs, 3,'Edgecoeffs', $
                   [2,3,4,5],[2,3])
  if cancel then return
  cancel = mc_cpar('mc_mkwavecalimgs1d',xranges, 4,'Xranges',[2,3,4,5],[1,2])
  if cancel then return
  cancel = mc_cpar('mc_mkwavecalimgs1d',slith_arc, 5,'Slith_arc',[2,3,4,5],0)
  if cancel then return

;  Get the number of orders

  norders = n_elements(xranges[0,*])

;  Set up some arrays
  
  wavecal = make_array(ncols,nrows,/DOUBLE,VALUE=!values.f_nan)
  spatcal = wavecal
  

  for i = 0,norders-1 do begin

;  Define some things
     
     start    = xranges[0,i]
     stop     = xranges[1,i]
     numwave  = fix(stop-start)+1
     xgrid    = findgen(numwave)+start
     y        = findgen(nrows)
     
;  Find the bottom and top of the slit
     
     botedge = mc_poly1d(xgrid,edgecoeffs[*,0,i])
     topedge = mc_poly1d(xgrid,edgecoeffs[*,1,i])
     dif     = topedge-botedge

;  Now get the pixel to arc conversions

     pixtoarc = fltarr(stop-start+1,2)
     pixtoarc[*,1] = float(slith_arc) / (topedge-botedge) 
     pixtoarc[*,0] = -1.* (pixtoarc[*,1] * botedge)

     for j =0,stop-start do begin
        
        wavecal[xgrid[j],floor(botedge[j]):ceil(topedge[j])] = wgrid[j]
        
        ypix = reform(y[floor(botedge[j]):ceil(topedge[j])])
        spix = mc_poly1d(ypix,reform(pixtoarc[j,*]))

        spatcal[xgrid[j],floor(botedge[j]):ceil(topedge[j])] = spix
        
     endfor
     
  endfor

end

