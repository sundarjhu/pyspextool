function mcshowordnum::init
  
  self.edgecoeffs = ptr_new(/allocate)
  self.xranges = ptr_new(/allocate)
  self.onums = ptr_new(/allocate)
  self.color = ptr_new(/allocate)

  return,1
  
end
;
;=============================================================================
;
pro mcshowordnum::set,xranges,edgecoeffs,onums,color

  *self.edgecoeffs = edgecoeffs
  *self.xranges = xranges
  *self.onums = string(onums,FORMAT='(I3)')
  *self.color = color

  return 
  
end
;
;=============================================================================
;
pro mcshowordnum::plot

  s = size(*self.edgecoeffs)
  norders = (s[0] eq 2) ? 1:s[3]
  
  
  for i = 0,norders-1 do begin
     
     xcen = ((*self.xranges)[1,i]+(*self.xranges)[0,i])/2.
     top = poly(xcen,(*self.edgecoeffs)[*,0,i])
     bot = poly(xcen,(*self.edgecoeffs)[*,1,i])

     ycen = (top+bot)/2.

     xyouts,xcen,ycen,(*self.onums)[i],COLOR=*self.color,ALIGNMENT=0.5
     
  endfor
     
  return
  
end
;
;=============================================================================
;
function mcshowordnum::cleanup

;-- free memory allocated to pointer when destroying object

  ptr_free,self.edgecoeffs
  ptr_free,self.xranges
  ptr_free,self.onums
  ptr_free,self.color
  
 return,1

end 

;
;=============================================================================
;
pro mcshowordnum__define
 
  void={mcshowordnum, $
        xranges:ptr_new(),$
        onums:ptr_new(),$
        color:ptr_new(),$
        edgecoeffs:ptr_new()}
  return 
 
end
