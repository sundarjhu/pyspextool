;+
; NAME:
;     mc_mkrectindcs
;
; PURPOSE:
;     To create interpolation indices for 1D extraction
;
; CALLING SEQUENCE:
;     result = mc_mkrectindcs1d(edgecoeffs,xranges,coeffs,slith_arc,ds,ystep,$
;                               XGRIDS=xgrids,SGRID=sgrid,UPDATE=update,$
;                               WIDGET_ID=widget_id,CANCEL=cancel)
;
; INPUTS:
;     edgecoeffs - Array [degree+1,2,norders] of polynomial coefficients 
;                  which define the edges of the orders.  array[*,0,i]
;                  are the coefficients of the bottom edge of the ith
;                  order and array[*,1,i] are the coefficients of the
;                  top edge of the ith order.
;     xranges    - An array [2] of pixel positions where the orders
;                  are completely on the array.
;     coeffs     - A structure with 1 or 2 tags.
;                  coeffs.c1coeffs = the polynomial coefficients
;                  giving the value of c1 as a function of column.
;                  coeffs.c2coeffs = the polynomial coefficients
;                  giving the value of c2 as a function of column.
;     slith_arc  - The slit height in arcseconds.
;     ds         - The sampling frequency in the angular (spatial)
;                  dimension in arcseconds.
;     ystep      - The number of pixels to step in the y direction
;                  when generating rectification indices.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     COEFF2INFO -
;     
;
; OUTPUTS:
;     result - 
;
; OPTIONAL OUTPUTS:
;     XGRIDS -
;     SGRID  - 
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     None
;
; DEPENDENCIES:
;     The Spextool package (and its dependencies) 
;
; PROCEDURE:
;
;
; EXAMPLES:
;
;
; MODIFICATION HISTORY:
;     2017-08-13 - Written by M. Cushing, University of Toledo
;     2019-03-29 - Removed the trimming of the order that was done in
;                  the 2D case as it caused problems with the order
;                  mask in later steps of the extraction.  
;-
function mc_mkrectindcs1d,edgecoeffs,xranges,coeffs,slith_arc,ds,ystep, $
                          XGRIDS=xgrids,SGRID=sgrid,UPDATE=update, $
                          WIDGET_ID=widget_id,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() lt 6 then begin
     
     print, 'Syntax - s = mc_mkinterpindxs(edgecoeffs,xranges,coeffs,$'
     print, '                              slith_arc,ds,XGRIDS=xgrids,$'
     print, '                              SGRID=sgrid,UPDATE=update,$'
     print, '                              WIDGET_ID=widget_id,CANCEL=cancel'
     cancel = 1
     return,-1
     
  endif
  
  cancel = mc_cpar('mc_mkinterpindxs',edgecoeffs,1,'Edgecoeffs',[4,5],[2,3])
  if cancel then return,-1

;  Get the Fanning showprogress widget running if need be.
  
  if keyword_set(UPDATE) and keyword_set(WIDGET_ID) then begin
        
     cancelbutton = (n_elements(WIDGET_ID) ne 0) ? 1:0
     progressbar = obj_new('SHOWPROGRESS',widget_id,COLOR=2,$
                           CANCELBUTTON=cancelbutton,$
                           MESSAGE='Generating rectification indices')
     progressbar -> start
     
  endif

;  How many orders

  s = size(xranges)
  norders = (s[0] eq 1) ? 1:s[2]
  
;  Determine y size of the output grid
  
  nsgrid = round(slith_arc/float(ds))+1
  sgrid  = findgen(nsgrid)*ds

;  Now start the loop over the orders

  for i = 0,norders-1 do begin

     start = (xranges[0,i])[0]
     stop  = (xranges[1,i])[0]
     
     nxgrid = stop-start+1
     xgrid  = dindgen(nxgrid) + start
     
     ix = dblarr(nxgrid,nsgrid,/NOZERO)
     iy = dblarr(nxgrid,nsgrid,/NOZERO)
     
     topslit = poly(xgrid,edgecoeffs[*,1,i])
     botslit = poly(xgrid,edgecoeffs[*,0,i])
     midslit = (topslit+botslit)/2D

;  Start the loop over the pixels

     for j = 0,nxgrid-1 do begin

        c1z = poly(xgrid[j],coeffs.c1coeffs)
        c1l = c1z
        c0l = xgrid[j]-c1l*midslit[j]
        linecoeffs = [c0l,c1l]

        if n_tags(coeffs) eq 2 then begin

           c2z = poly(xgrid[j],coeffs.c2coeffs)
           c1z = poly(xgrid[j],coeffs.c1coeffs)

           c2l = c2z
           c1l = c1z-2*c2l*midslit[j]
           c0l = xgrid[j]-c1l*midslit[j]-c2l*midslit[j]^2
           linecoeffs =  [c0l,c1l,c2l]
                      
        endif

;  Now compute the intercepts, top intercept first
        
        dif = xgrid-poly(topslit,linecoeffs)
        linterp,dif,xgrid,0,xtop
        ytop = poly(xtop,edgecoeffs[*,1,i])
     
;  Now the bottom intercept
     
        dif = xgrid-poly(botslit,linecoeffs)
        linterp,dif,xgrid,0,xbot
        ybot = poly(xbot,edgecoeffs[*,0,i])
        
;  Now determine the length of the line in pixels        

        qtrap,'mc_polyarclenfunc',ybot,ytop,linelength,COEFF=linecoeffs        
        
;  Now integrate up from the bottom of the slit along the line in
;  units of pixels

        k = 1
        sline = 0D
        yline = ybot

        while ybot+k*ystep lt ytop do begin

           qtrap,'mc_polyarclenfunc',ybot,ybot+k*ystep,val,COEFF=linecoeffs
           sline = [sline,val]
           yline = [yline,ybot+k*ystep]
           k++

        endwhile

;  Tack on the top of the slit
        
        sline = [sline,linelength]
        yline = [yline,ytop]

;  Normalize by the slit length in arcseconds
        
        sline = sline/linelength*slith_arc     
        
;  Now figure out the x and y positons on sgrid 

        nyline = interpol(yline,sline,sgrid)

;  Store the results
        
        iy[j,*] = nyline
        ix[j,*] = poly(nyline,linecoeffs)

  endfor

     
;;  Now trim to avoid the left and the right of the order
;  
;  xmask = (ix le (xranges[0,i]+2)) + (ix ge (xranges[1,i]-2))
;  xmask = total(xmask,2)
;  zx = where(xmask eq 0,xcnt)
;  xgood = [zx[0],zx[xcnt-1]]
;  xgrid = xgrid[xgood[0]:xgood[1]]
;  ix = ix[xgood[0]:xgood[1],*]
;  iy = iy[xgood[0]:xgood[1],*]
 
;  Store the results in a structure

     data = [[[ix]],[[iy]]]
     tag = string(i)

     if i eq 0 then begin

        s = create_struct(tag,data)
        xgrids = create_struct(tag,xgrid)
        
     endif else begin

        s = create_struct(s,tag,data)
        xgrids = create_struct(xgrids,tag,xgrid)
        
     endelse
         
;  Do the update stuff
     
     if keyword_set(UPDATE) then begin
        
        if keyword_set(WIDGET_ID) then begin
           
           if cancelbutton then begin
              
              cancel = progressBar->CheckCancel()
              if cancel then begin
                 
                 progressBar->Destroy
                 obj_destroy, progressbar
                 cancel = 1
                 return,-1
                 
              endif
              
           endif
           percent = (i+1)*(100./float(norders))
           progressbar->update, percent
           
        endif else begin

         if norders gt 1 then mc_loopprogress,i,0,norders-1
           
        endelse
        
     endif
                
  endfor
  
  if keyword_set(UPDATE) and keyword_set(WIDGET_ID) then begin
     
     progressbar-> destroy
     obj_destroy, progressbar
     
  endif 

  return, s
  

end
  
