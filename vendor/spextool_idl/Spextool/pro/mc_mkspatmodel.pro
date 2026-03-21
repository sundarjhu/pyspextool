;+
; NAME:
;
;
; PURPOSE:
;
;
; CALLING SEQUENCE:
;
;
; INPUTS:
;
;
; OPTIONAL INPUTS:
;
;
; KEYWORD PARAMETERS:
;
;
; OUTPUTS:
;
;
; OPTIONAL OUTPUTS:
;
;
; COMMON BLOCKS:
;
;
; RESTRICTIONS:
;
;
; DEPENDENCIES:
;
;
; PROCEDURE:
;
;
; EXAMPLES:
;
;
; MODIFICATION HISTORY:
;
;-
function mc_mkspatmodel,rectorder,prof,tracecoeffs,apradii,AVEPROF=aveprof, $
                        ATMOS=atmos,DEBUG=debug,CANCEL=cancel

  cancel = 0

  ; Parse the rectorder array

  sgrid = reform(rectorder[0,1:*])
  xgrid = reform(rectorder[1:*,0])
  rectorder = rectorder[1:*,1:*]

;  Get sizes
  
  s = size(rectorder,/DIMEN)
  nx = s[0]
  ny = s[1]
  naps = n_elements(tracecoeffs[0,*])

;  Create the 1D profile coefficients 

  ndeg = 2
  coeffs = fltarr(ndeg+1,ny)

  if keyword_set(AVEPROF) then begin
     
     coeffs[0,*] = prof   

  endif else begin

;  Set zero values in the profile to NaN

     z = where(prof eq 0,cnt)
     if cnt ne 0 then prof[z] = !values.f_nan

;  Create the mean profile map and divide into the rectorder
     
     meanmap = rebin(reform(prof,1,ny),nx,ny)
     ratio = rectorder/meanmap

;  Determine the pixels with which to get the normalization values

     trace_arc = fltarr(nx,naps)
     for i = 0,naps-1 do trace_arc[*,i] = poly(xgrid,tracecoeffs[*,i])
     trace_arc = median(trace_arc,DIMEN=1,/EVEN)

     slitmask = mc_mkapmask(sgrid,trace_arc,apradii,CANCEL=cancel)
     if cancel then return, -1
     z = where(slitmask eq 0)
     slitmask[z] = !values.f_nan
     slitmask = rebin(reform(slitmask,1,ny),nx,ny)

;  Now generate the normalization values

     norms = median(ratio*slitmask,/EVEN,DIMENSION=2)

;  Normalize the order
     
     z = where(norms eq 0.0 or norms eq -0.0,cnt)
     if cnt ne 0 then norms[z] = !values.f_nan
     rectorder = temporary(rectorder)/rebin(norms,nx,ny)  

     if debug then begin
        
        ximgtool,rectorder
        cancel = mc_pause()
        print, cancel
        if cancel then return, -1
        
     endif
       
;  Now fit each row with a polynomial

     if keyword_set(ATMOS) then begin

        linterp,atmos.awave,atmos.atrans,xgrid,natrans,MISSING=1
        natrans = natrans > 0.01
        yerr = 1/natrans
        
     endif
     
     for i = 0,ny-1 do begin
        
        c = mc_robustpoly1d(xgrid,rectorder[*,i],ndeg,3.5,0.01,YERR=yerr, $
                            /GAUSSJ,/SILENT,OGOODBAD=ogoodbad,CANCEL=cancel)
        if cancel then c = replicate(!values.f_nan,ndeg+1)
        
        coeffs[*,i] = c
        
;        if debug then begin
;           
;           mc_moments,rectorder[*,i],mean,var,stddev,ROBUST=4,/SILENT
;           yrange = [mean-10*stddev,mean+10*stddev]
;           
;           plot,xgrid,rectorder[*,i],/XSTY,/YSTY,YRANGE=yrange,PSYM=1,$
;                TITLE='Row '+strtrim(sgrid[i],2)
;           oplot,xgrid,poly(xgrid,c),COLOR=2
;           z = where(ogoodbad eq 0,cnt)
;           if cnt ne 0 then begin
;              
;              oplot,xgrid[z],rectorder[z,i],COLOR=4,PSYM=1
;              
;           endif
;           cancel = mc_pause()
;           if cancel then return,-1
;           
;        endif
        
     endfor

  endelse

; Create the spatial model
  
  spatmod = fltarr(nx,ny,/NOZERO)
  for i = 0,ny-1 do spatmod[*,i] = poly(xgrid,coeffs[*,i])

  if debug then begin
  
     ximgtool,spatmod
     cancel =mc_pause()
     if cancel then return, -1


  endif
  
  return, {model:spatmod,sgrid:sgrid}
  
end
