function mcoplotedge::init
  
  self.edgecoeffs = ptr_new(/allocate)
  self.xranges = ptr_new(/allocate)
  self.orders = ptr_new(/allocate)

  return,1
  
end
;
;=============================================================================
;
pro mcoplotedge::set,xranges,edgecoeffs,ORDERS=orders

  *self.edgecoeffs = edgecoeffs
  *self.xranges = xranges
  if n_elements(orders) ne 0 then *self.orders = orders

  return 
  
end
;
;=============================================================================
;
pro mcoplotedge::plot

  s = size(*self.edgecoeffs)
  norders = (s[0] eq 2) ? 1:s[3]
  

  for i = 0,norders-1 do begin
     
     del = (*self.xranges)[1,i]-(*self.xranges)[0,i]+1
     x = findgen(del)+(*self.xranges)[0,i]

     
     oplot,x,poly(x,(*self.edgecoeffs)[*,0,i]),COLOR=13
     oplot,x,poly(x,(*self.edgecoeffs)[*,1,i]),COLOR=13

     if n_elements(*self.orders) ne 0 then begin

        xcen = (*self.xranges)[0,i];+(*self.xranges)[0,i])/2.
        top = poly(xcen,(*self.edgecoeffs)[*,0,i])
        bot = poly(xcen,(*self.edgecoeffs)[*,1,i])
        ycen = (top+bot)/2.

        xyouts,xcen,ycen,strtrim((*self.orders)[i],2),COLOR=5,ALIGNMENT=0

     endif
     
  endfor
     
  return
  
end
;
;=============================================================================
;
function mcoplotedge::cleanup

;-- free memory allocated to pointer when destroying object

 ptr_free,self.edgecoeffs
 ptr_free,self.xranges
 ptr_free,self.orders

 return,1

end 

;
;=============================================================================
;
pro mcoplotedge__define
 
  void={mcoplotedge, $
        xranges:ptr_new(),$
        edgecoeffs:ptr_new(),$
        orders:ptr_new()}
  return 
 
end
