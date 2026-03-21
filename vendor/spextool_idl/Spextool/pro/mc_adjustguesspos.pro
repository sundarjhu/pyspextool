;+
; NAME:
;     mc_adjustguesspos
;
; PURPOSE:
;    To adjust the guess positions via a cross correlation
;
; CALLING SEQUENCE:
;     result = mc_adjustguesspos(edgecoeffs,xranges,flat,omask,orders,$
;                                ycororder,ybuffer,[offset],DEFAULT=default,$
;                                CANCEL=cancel)
;
; INPUTS:
;     edgecoeffs - Array [degree+1,2,norders] of polynomial coefficients 
;                  which define the edges of the orders.  array[*,0,0]
;                  are the coefficients of the bottom edge of the
;                  first order and array[*,1,0] are the coefficients 
;                  of the top edge of the first order.
;     xranges    - An array [2,norders] of column numbers where the
;                  orders are completely on the array
;     flat       - An array [ncols,nrows] of a raw flat field.
;     omask      - An array [ncols,nrows] where each pixel is set to
;                  its order number.  Interorder pixels are set to
;                  zero.
;     orders     - A vector [norders] giving the order numbers.
;     ycororder  - The order with which to do the cross correlation
;     ybuffer    - The number of pixels to buffer from the top and
;                  bottom of the array.
;
; OPTIONAL INPUTS:
;     offset - If given, no offset will be computed but instead the
;              value passed will be used.
;
; KEYWORD PARAMETERS:
;     DEFAULT - Set to simply return the default guess positions and xranges
;     CANCEL  - Set on return if there is a problem.
;
; OUTPUTS:
;     A 2-tag structure.  result.guesspos = the adjusted guess
;     positions.  result.xranges = the adusted xranges
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
;     Performed a vertical cross correlation between the order mask
;     and the raw flat field of a single order to identify the shift
;     between the two.  This shift is subtracted from the guess
;     positions.  The xranges are then checked to ensure the order
;     range does not fall off the array.
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2018-01-12 - Written by M. Cushing, University of Toledo
;     2018-04-30 - Added the DEFAULT keyword.
;     2019-04-14 - Adjust the number of shifts from 2*slitw_pix to
;                  1.8*slitw_pix because it was picking up a different
;                  order in iSHELL L1.
;     2019-08-21 - Added the offset parameter.
;-
function mc_adjustguesspos,edgecoeffs,xranges,flat,omask,orders,ycororder, $
                           ybuffer,offset,DEFAULT=default,CANCEL=cancel

  cancel = 0

  ;  Check parameters

  if n_params() lt 7 then begin
     
     print, 'Syntax - result = mc_adjustguesspos(edgecoeffs,xranges,flat,$'
     print, '                                    omask,orders,ycororder, $'
     print, '                                    ybuffer,[offset],$'
     print, '                                    DEFAULT=default,CANCEL=cancel)'
     cancel = 1
     return,-1
     
  endif
  cancel = mc_cpar('mc_adjustguesspos',edgecoeffs,1,'Edgecoeffs',[4,5],[2,3])
  if cancel then return,-1
  cancel = mc_cpar('mc_adjustguesspos',xranges,2,'Xranges',[2,3],[1,2])
  if cancel then return,-1
  cancel = mc_cpar('mc_adjustguesspos',flat,3,'Flat',[2,3,4,5],2)
  if cancel then return,-1  
  cancel = mc_cpar('mc_adjustguesspos',omask,4,'Omask',[2,3],2)
  if cancel then return,-1
  cancel = mc_cpar('mc_adjustguesspos',orders,5,'Orders',[2,3],1)
  if cancel then return,-1  
  cancel = mc_cpar('mc_adjustguesspos',ycororder,6,'YCororder',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_adjustguesspos',ybuffer,7,'Ybuffer',[2,3],0)
  if cancel then return,-1      


;  Get pertinent information
  
  norders = n_elements(xranges[0,*])
  s = size(omask,/DIMEN)
  ncols = s[0]
  nrows = s[1]

;  Compute the guess positions
  
  guesspos = intarr(2,norders)
  for i = 0,norders-1 do begin
     
     x       = total(xranges[*,i])/2.
     botedge = mc_poly1d(x,edgecoeffs[*,0,i])
     topedge = mc_poly1d(x,edgecoeffs[*,1,i])

     guesspos[*,i] = round([x,total([botedge,topedge])/2.])
     
  endfor

  if keyword_set(DEFAULT) then return, {guesspos:guesspos,xranges:xranges}
  
;  Now do the y cross correlation 

  if n_params() eq 7 then begin
  
;  Set all orders other than ycororder to zero.     

     z = where(omask ne ycororder,COMP=good)
     omask[z] = 0
     omask[good] = 1
     
;  Find the maximum slit height (pixels) for the ycororder
     
     z = where(orders eq ycororder)
     
     x = indgen(xranges[1,z]-xranges[0,z]+1)+total(xranges[0,z])     
     botedge = mc_poly1d(x,edgecoeffs[*,0,z])
     topedge = mc_poly1d(x,edgecoeffs[*,1,z])
     
     slith_pix = ceil(max(topedge-botedge))
     
; Determine the top and bottom row of the subimage to clip out
     
     topidx = round(max(topedge)+slith_pix)
     botidx = round(min(botedge)-slith_pix)
     
     subflat  = flat[*,botidx:topidx]
     subomask = omask[*,botidx:topidx]
     s = size(subflat,/DIMEN)
     
;  Let's get the shifts set up 
     
     nshifts = slith_pix*1.8+1
     shifts = indgen(nshifts)-nshifts/2.
     sum = fltarr(nshifts)
     
;  Do it
     
     for i = 0,nshifts-1 do begin
        
        hbot = -shifts[i] > 0
        htop = ((s[1]-1)-shifts[i]) < (s[1]-1)
        
        mbot = shifts[i] > 0
        mtop = ((s[1]-1)+shifts[i]) < (s[1]-1)
        
        sum[i] = total(subflat[*,hbot:htop]*subomask[*,mbot:mtop])
        
     endfor
     
     junk = max(sum,idx)
     offset = shifts[idx]

  endif 
          
;  Now check the xranges

  nxranges = intarr(2,norders,/NOZERO)
  for i = 0,norders-1 do begin

     x = indgen(xranges[1,i]-xranges[0,i]+1)+fix(total(xranges[0,i]))
     botedge = mc_poly1d(x,edgecoeffs[*,0,i])-offset
     topedge = mc_poly1d(x,edgecoeffs[*,1,i])-offset

     z = where(botedge gt ybuffer-1 and topedge lt ncols-ybuffer-1,cnt)
     nxranges[*,i] = [min(x[z],MAX=max),max]

  endfor

  guesspos[1,*] = guesspos[1,*]-offset

  return, {guesspos:guesspos,xranges:nxranges}

end
