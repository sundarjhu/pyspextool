;+
; NAME:
;     mc_fitlines2dxd
;
; PURPOSE:
;     To fit emissions lines identified by mc_findlines2dxd.
;
; CALLING SEQUENCE:
;     result = mc_fitlines2dxd(lxy,lorders,edgecoeffs,xranges,orders,$
;                              linedeg,LINEFITQAPLOT=linefitqaplot, $
;                              LINESUMQAPLOT=linesumqaplot,CANCEL=cancel)
;
; INPUTS:
;     lxy        - A structure with [nlines] tags where each tag
;                  contains the x,y positions of a single line.  They
;                  are  given as an array [[x],[y]].  Output from
;                  mc_findlines2dxd.pro.
;     lorders    - A [nlines] array giving the order number of each
;                  line.  Output from mc_findlines2dxd.pro.
;     edgecoeffs - Array [degree+1,2,norders] of polynomial coefficients 
;                  which define the edges of the orders.  array[*,0,i]
;                  are the coefficients of the bottom edge of the ith
;                  order and array[*,1,i] are the coefficients of the
;                  top edge of the ith order.
;     xranges    - An array [2,norders] of pixel positions where the orders
;                  are completely on the array.
;     orders     - An [norders] array giving the order numbers.
;     linedeg    - The polynomial degree of the fit to the line, 1 or 2.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     LINEFITQAPLOT - If given, a Quality Assurance plot of the line
;                     fits will be generated.  LINEFITQAPLOT is a
;                     structure with the following tags:
;
;
;     LINESUMQAPLOT - If given, a Quality Assurance plot of the line
;                     fits will be generated.  LINEFITQAPLOT is a
;                     structure with the following tags:
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;
;
; DEPENDENCIES:
;     Spextool package (and its dependencies)
;
; PROCEDURE:
;     NA
;
; EXAMPLES:
;     Eh
;
; MODIFICATION HISTORY:
;
;-
;
; =============================================================================
;
pro fitlines2dxd_plotarraypoints,ncols,nrows,xvals,yvals,zvals,title, $
                                cbartitle,orders,edgecoeffs,xranges
  norders = n_elements(orders)
  
  pos = mc_gridplotpos([0.3,0.3,0.95,0.90],[1,1],0.02,/SQUARE)
  pos[0] = pos[0]-0.1
  pos[2] = pos[2]-0.1

  ratio = nrows/float(ncols)
  if ratio ne 1 then begin
     
     if ratio lt 1 then pos[3] = pos[3]-(pos[3]-pos[1])*ratio
     if ratio gt 1 then pos[2] = pos[2]-(pos[2]-pos[0])/ratio
     
  endif
  
  plot,[1],[1],/NODATA,XRANGE=[0,ncols],YRANGE=[0,nrows],FONT=0, $
       POSITION=pos[*,0],XTITLE='Column (pixels)', $
       YTITLE='Row (pixels)',/XSTY,/YSTY,TITLE=title
  
  mc_mkct,13,BOT=cbot
  
  ncolors = !d.n_colors < 255
  ncolors = ncolors-cbot

  if ~keyword_set(GOODBAD) then $
     goodbad = make_array(n_elements(xvals),VALUE=1,/INT)

  zbad = where(goodbad eq 0,cnt_bad,COMP=zgood,NCOMP=cnt_good)
     
  byt_zvalss = bytscl(zvals,TOP=ncolors-1,MIN=min(zvals[zgood]),$
                      MAX=max(zvals[zgood])) + cbot
  
  plotsym,0,0.9,/FILL
  plots,xvals,yvals,PSYM=8
  plotsym,0,0.7,/FILL
  for i = 0,n_elements(xvals)-1 do plots,xvals[i],yvals[i],PSYM=8, $
                                         COLOR=byt_zvalss[i]

  if cnt_bad ne 0 then plots,xvals[zbad],yvals[zbad],PSYM=6,SYMSIZE=1.5

;  Plot orders

  if n_params() gt 7 then begin
  
     for i = 0,norders-1 do begin
        
        x = findgen(xranges[1,i]-xranges[0,i]+1)+xranges[0,i]        
        topslit = poly(x,edgecoeffs[*,1,i])
        botslit = poly(x,edgecoeffs[*,0,i])
        !p.thick = 1
        oplot,x,botslit
        oplot,x,topslit
        
        midx = total(xranges[1,i]+xranges[0,i])/2.
        midy = total(poly(midx,edgecoeffs[*,1,i])+ $
                     poly(midx,edgecoeffs[*,0,i]))/2.
        xyouts,midx,midy,strtrim(orders[i]),FONT=0,ALIGNMENT=0.5,CHARSIZE=0.7
        
     endfor
     !p.thick = 3
     
  endif

;  Make the color bar

  min = min(zvals,MAX=max)
  npoints = 20
  del = (max-min)/float(npoints)
  
  cb = rebin(reform(findgen(npoints)*del+min,1,npoints),2,npoints)
  byt_cb = bytscl(cb,TOP=ncolors-1) + cbot
  poss = [pos[2]+0.12,pos[1],pos[2]+0.12+0.05,pos[3]]
  
  tvimage,byt_cb,POSITION=poss
  plot,[1],[1],/NODATA,FONT=0,/NOERASE,$
       POSITION=poss,$
       YRANGE=[min(zvals),max(zvals)],/XSTY,XTICKS=1,$
       XRANGE=[0,max],YSTY=1,$
       XTICKNAME=replicate(' ',10), $
       YTICKLEN=0.2,$
       YTITLE=cbartitle

;  Do the marginalizations

;  Bottom first
  
  offset = 0.08
  width = 0.17
  posbot = [pos[0],pos[1]-offset-width,pos[2],pos[1]-offset]

  plot,[1],[1],POSITION=posbot,/NOERASE,FONT=0,$
       XRANGE=[0,ncols],/YSTY,$
       YRANGE=mc_bufrange([min(zvals),max(zvals)],0.1),/XSTY, $
       XTICKNAME=replicate(' ',10),YTICKNAME=replicate(' ',10)


  plotsym,0,0.9,/FILL
  plots,xvals,zvals,PSYM=8
  plotsym,0,0.7,/FILL
  for i = 0,n_elements(xvals)-1 do plots,xvals[i],zvals[i],PSYM=8, $
                                         COLOR=byt_zvalss[i]

  if cnt_bad ne 0 then plots,xvals[zbad],zvals[zbad],PSYM=6,SYMSIZE=1.5
  
;  Next the left
    
  offset = 0.08
  width = width*float(!d.y_vsize)/!d.x_vsize
  posleft = [pos[0]-offset-width,pos[1],pos[0]-offset,pos[3]]

  plot,[1],[1],POSITION=posleft,/NOERASE,FONT=0,$
       YRANGE=[0,nrows],/YSTY,$
       XRANGE=mc_bufrange([max(zvals),min(zvals)],0.1),/XSTY, $
       YTICKNAME=replicate(' ',10),XTICKNAME=replicate(' ',10)

  plotsym,0,0.9,/FILL
  plots,zvals,yvals,PSYM=8
  plotsym,0,0.7,/FILL
  for i = 0,n_elements(xvals)-1 do plots,zvals[i],yvals[i],PSYM=8, $
                                         COLOR=byt_zvalss[i]

  if cnt_bad ne 0 then plots,zvals[zbad],yvals[zbad],PSYM=6,SYMSIZE=1.5
  
end
;
;===============================================================================
;
pro fitlines2dxd_plotlinefit,img,ncols,nrows,x,topslit,botslit,delslit,lx,ly, $
                             fwhm,linecoeffs,goodbad,xtop,ytop,xbot,ybot, $
                             xmid,ymid,order,wavelength,id

  mc_mkct,0,BOT=cbot
  
  xsize = fwhm*10
  ysize = max(delslit)+10
           
;  Locate the order

  subimg = mc_snipimgc(img,xmid,ymid,xsize,ysize,dncols,dnrows, $
                       xrange,yrange,CANCEL=cancel)
  if cancel then return

  nyl = 0.12
  nyu = 0.95
  nxl = 0.1
  nxu = 0.25
;  nxu = 0.1+(nyu-nyl)*float(xsize)/ysize*8.5/11
  position = [nxl,nyl,nxu,nyu]

;  nxu =0.3
  
  mc_imgrange,subimg,min,max,/ZSCALE
;  tv,mc_bytsclimg(subimg,0,cbot,MIN=min,MAX=max),nxl,nyl,/NORMAL
  tvimage,mc_bytsclimg(subimg,0,cbot,MIN=min,MAX=max), $
          POSITION=position,/NOINTERP,/KEEP_ASPECT_RATIO
  
  plot,[1],[1],/NODATA,/NOERASE,XSTY=1,YSTY=1, $
       XRANGE=[xrange[0]-0.5,xrange[1]+0.5], $
       YRANGE=[yrange[0]-0.5,yrange[1]+0.5],XMARGIN=[0,0], $
       YMARGIN=[0,0],NOCLIP=0,POSITION=position, $
       XMINOR=1,FONT=0,XTICKS=1,XTICKLEN=-0.01,XTITLE='Column',YTITLE='Row',$
       YTICKLEN=-0.1
  
  oplot,x,topslit,COLOR=13,THICK=4
  oplot,x,botslit,COLOR=13,THICK=4

  plotsym,0,0.5,/FILL
  plots,lx,ly,PSYM=8,COLOR=3
  oplot,poly(ly,linecoeffs),ly,COLOR=2
  z = where(goodbad eq 0,cnt)
  if cnt ne 0 then oplot,lx[z],ly[z],PSYM=8,COLOR=4           
  
  plotsym,0,1.5,/FILL
  plots,xtop,ytop,PSYM=8,COLOR=6
  plots,xbot,ybot,PSYM=8,COLOR=6
  plots,xmid,ymid,PSYM=8,COLOR=6

  pos = mc_gridplotpos([nxu+0.15,nyl,0.95,nyu-0.1],[1,2],0.01)
  
  plot,ly,lx,/NODATA,/NOERASE,POSITION=pos[*,0],FONT=0,/XSTY,/YSTY,$
       XRANGE=mc_bufrange([ybot,ytop],0.05),$
       YRANGE=mc_bufrange([xtop,xbot],0.05),$
       YTITLE='Column',XTICKNAME=replicate(' ',10)
           
  plotsym,0,0.8,/FILL
  oplot,ly,lx,PSYM=8
  plotsym,0,0.6,/FILL
  oplot,ly,lx,PSYM=8,COLOR=3
  oplot,ly,poly(ly,linecoeffs),COLOR=2
  z = where(goodbad eq 0,cnt)
  if cnt ne 0 then oplot,ly[z],lx[z],PSYM=8,COLOR=4

  plotsym,0,1.5,/FILL
  plots,ytop,xtop,PSYM=8,COLOR=6
  plots,ybot,xbot,PSYM=8,COLOR=6
  plots,ymid,xmid,PSYM=8,COLOR=6
  
  resid = lx-poly(ly,linecoeffs)
  plot,ly,resid,/NODATA,/NOERASE,POSITION=pos[*,1],FONT=0,/XSTY, $
       /YSTY,XRANGE=mc_bufrange([ybot,ytop],0.05),$
       YRANGE=mc_bufrange([min(resid,MAX=max),max],0.05),$
       YTITLE='Residuals (pixels)',XTITLE='Row'
           
  plotsym,0,0.8,/FILL
  oplot,ly,resid,PSYM=8
  plotsym,0,0.6,/FILL
  oplot,ly,resid,PSYM=8,COLOR=3
  plots,!x.crange,[0,0],LINESTYLE=1
  z = where(goodbad eq 0,cnt)
  if cnt ne 0 then oplot,ly[z],resid[z],PSYM=8,COLOR=4           
  
  label = 'Order '+strtrim(order,2)
  xyouts,pos[0,0],nyu-0.02,label,/NORM,FONT=0,ALIGNMENT=0
  
  label = 'Column='+strtrim(xmid,2)+ $
          ', !9l!X='+strtrim(wavelength,2)+' !9m!Xm'

  if strlen(strtrim(id,2)) ne 0 then label = label+', '+strtrim(id,2)
  
  xyouts,pos[0,0],nyu-0.05,label,/NORM,FONT=0,ALIGNMENT=0
  
  label = '!9D!Xx='+strtrim(xtop-xbot,2)
  xyouts,pos[0,0],nyu-0.08,label,/NORM,FONT=0,ALIGNMENT=0
  
end
;
;==============================================================================
;
pro fitlines2dxdxd_plotmarginals,ncols,nrows,xvals,yvals,zvals, $
                                ytitle,orders,ovals,CANCEL=cancel

  cancel = 0
  
  pos = mc_gridplotpos([0.1,0.1,0.85,0.95],[1,2],0.05)
  
  min = min(zvals,MAX=max)
  yrange = mc_bufrange([min,max],0.1)
  plot,[1],[1],XRANGE=[0,ncols-1],YRANGE=yrange,/XSTY,/YSTY,/NODATA, $
       POS=pos[*,0],FONT=0,XTITLE='Column (pixels)', YTITLE=ytitle

  norders = n_elements(orders)
  if n_params() gt 6 and n_elements(orders) ne 1 then begin
     
     mc_ldcolortempct,norders,/INVERT
     
     for i = 0,norders-1 do begin
        
        z = where(fix(ovals) eq orders[i],cnt)
        if cnt ne 0 then begin
           
           plotsym,0,0.9,/FILL
           oplot,xvals[z],zvals[z],PSYM=-8
           plotsym,0,0.7,/FILL
           oplot,xvals[z],zvals[z],COLOR=2+i,PSYM=8
           
        endif
     
     endfor

     cb = rotate(rebin(findgen(norders)+2,norders,20),1)
     poss = [pos[2,1]+0.08,pos[1,1],pos[2,1]+0.11,pos[3,0]]
     tvimage,cb,POSITION=poss
     plot,[1],[1],/NODATA,FONT=0,/NOERASE,$
          POSITION=poss,$
          YRANGE=[min(orders),max(orders)],/XSTY,XTICKS=1,$
          XRANGE=[0,max],YSTY=1,$
          XTICKNAME=replicate(' ',10), $
          YTICKLEN=0.2,$
          YTITLE='Order Number'
     
  endif else begin

     plotsym,0,0.7,/FILL
     oplot,xvals,zvals,PSYM=8

  endelse

  plot,[1],[1],XRANGE=[0,nrows],$
       YRANGE=yrange,/XSTY,/YSTY,/NODATA,POS=pos[*,1],$
       FONT=0,/NOERASE,XTITLE='Row (Pixels)',YTITLE=ytitle

  if n_params() gt 6 and n_elements(orders) ne 1 then begin  
  
     for i = 0,norders-1 do begin
        
        z = where(fix(ovals) eq orders[i],cnt)
        if cnt ne 0 then begin
           
           plotsym,0,0.7,/FILL
           oplot,yvals[z],zvals[z],COLOR=2+i,PSYM=-8
           plotsym,0,0.9,/FILL
           oplot,yvals[z],zvals[z],PSYM=8
           plotsym,0,0.7,/FILL
           oplot,yvals[z],zvals[z],COLOR=2+i,PSYM=8
           
        endif
        
     endfor

  endif else begin
     
     plotsym,0,0.7,/FILL
     oplot,yvals,zvals,PSYM=8

  endelse
       
end
;
;===============================================================================
;
function mc_fitlines2dxd,lxy,lorders,edgecoeffs,xranges,orders,linedeg, $
                         LINEFITQAPLOT=linefitqaplot, $
                         LINESUMQAPLOT=linesumqaplot,CANCEL=cancel
  
  cancel = 0

;  Check parameters 

  if n_params() lt 6 then begin

     print, ' Syntax - result = mc_fitlines2dxd(lxy,lorders,edgecoeffs,$'
     print, '                                   xranges,orders,linedeg, $'
     print, '                                   LINEFITQAPLOT=linefitqaplot,$'
     print, '                                   LINESUMQAPLOT=linesumqaplot,$'
     print, '                                   CANCEL=cancel)'
     
     cancel = 1
     return,-1
     
  endif
  cancel = mc_cpar('mc_fitlines2dxd',lxy,1,'Lxy',[8],1)
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines2dxd',lorders,2,'Lorders',[2,3],1)
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines2dxd',edgecoeffs,3,'Edgecoeffs',[2,3,4,5],[2,3])
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines2dxd',xranges,4,'Xranges',[2,3,4,5],[1,2])
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines2dxd',orders,5,'Orders',[2,3],1)
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines2dxd',linedeg,6,'Linedeg',[2,3],0)
  if cancel then return,-1
     
;  Get array sizes
  
  s = size(edgecoeffs)
  norders = n_elements(orders)
  nlines = n_tags(lxy)

;  Initialize the output structure

  struc = {linecoeffs:dblarr(linedeg+1),xbot:0D,ybot:0D,xmid:0D,ymid:0D, $
           xtop:0D,ytop:0D,order:100,deltax:0D,slope:0D,angle:0D}
  lstruc = [struc]

;  Unpack QA plot information if necessary

  if keyword_set(LINEFITQAPLOT) then begin

     img = linefitqaplot.img

     s = size(img,/DIMEN)
     ncols = s[0]
     nrows = s[1]

     prefix = linefitqaplot.prefix

     waves = linefitqaplot.waves

     ids = linefitqaplot.ids

     fwhms = linefitqaplot.fwhms
     
     !x.thick=3
     !y.thick=3
     !p.thick=3
     
     mc_mkct,0,BOT=cmapbot
     mc_setpsdev,prefix+'_2DLineFits.ps',11,8.5,/LANDSCAPE,FONT=13,BITS=8
     
  endif
  
;  Loop over each order
  
  for i = 0,norders-1 do begin

;  Check to see if there are any lines
     
     zlines = where(lorders eq orders[i],cnt)

     if cnt eq 0 then continue

;  Now the start the process
     
     x = findgen(xranges[1,i]-xranges[0,i]+1)+xranges[0,i]
     
;  Compute properties of this order
     
     topslit = poly(x,edgecoeffs[*,1,i])
     botslit = poly(x,edgecoeffs[*,0,i])
     midslit = (topslit+botslit)/2D
     delslit = topslit-botslit
     slith_pix = max(delslit)

;  Now loop over the lines in this order
     
     for j = 0,cnt-1 do begin
        
        lx = (lxy.(zlines[j]))[*,0]
        ly = (lxy.(zlines[j]))[*,1]
        
;  Fit the line 
        
        linecoeffs = mc_robustpoly1d(ly,lx,linedeg,3,0.01,/SILENT, $
                                     OGOODBAD=ogoodbad1,YFIT=test,CANCEL=cancel)
        if cancel then return,-1
        
;  Get intercepts of line and edges of the order.
        
;  Top intercept
        
        z = where(x ge min(lx)-10 and x le max(lx)+10)
        
        xline = poly(topslit[z],linecoeffs)
        dif = x[z]-xline
        linterp,dif,x[z],0,xtop
        ytop = (poly([xtop],edgecoeffs[*,1,i]))[0]
        
;  Bottom intercept
        
        xline = poly(botslit[z],linecoeffs)
        dif = x[z]-xline
        linterp,dif,x[z],0,xbot
        ybot = (poly([xbot],edgecoeffs[*,0,i]))[0]

;  Then get the intercept of the line with the geometric midslit

        xline = poly(midslit[z],linecoeffs)
        dif = x[z]-xline
        linterp,dif,x[z],0,xmid
        linterp,x,midslit,xmid,ymid

;  get QA delta X, delta Y, and slope of the line

        deltax = poly(ytop,linecoeffs)-poly(ybot,linecoeffs)
        deltay = ytop-ybot
        slope  = deltax/deltay

;  Store the results

        lstruc = [lstruc,{linecoeffs:double(linecoeffs), $
                          xbot:double(xbot),ybot:double(ybot), $
                          xmid:double(xmid),ymid:double(ymid), $
                          xtop:double(xtop),ytop:double(ytop), $
                          order:fix(orders[i]),deltax:deltax,slope:slope, $
                          angle:atan(slope)*180/!pi}]               
        
;  Plot the QA plot if requested
        
        if keyword_set(LINEFITQAPLOT) then begin

           fitlines2dxd_plotlinefit,img,ncols,nrows,x,topslit,botslit,delslit, $
                                    lx,ly,fwhms[zlines[j]],linecoeffs, $
                                    ogoodbad1,xtop,ytop,xbot,ybot,xmid,ymid, $
                                    orders[i],waves[zlines[j]],ids[zlines[j]]
           erase
           
        endif
        
     endfor
     
  endfor
  
;  Close out the LineFit QA plot
  
  if keyword_set(LINEFITQAPLOT) then mc_setpsdev,/CLOSE,/CONVERT,/ERASE
  
;  Clip the first blank structure
  
  lstruc = lstruc[1:*]
  
;   Now do the Line Summary QA plot

  if keyword_set(LINESUMQAPLOT) then begin

     ncols = linesumqaplot.ncols
     nrows = linesumqaplot.nrows
     prefix = linesumqaplot.prefix

     s = lstruc
         
     deltax = s.deltax
     deltay = s.ytop-s.ybot
     slope = s.deltax/deltay
     angle = s.angle
     xx = s.xmid
     yy = s.ymid
     oo = s.order

     mc_setpsdev,prefix+'_2DLineSummary.ps',11,8.5,/LANDSCAPE,FONT=13,BITS=8
     
     mc_mkct
     mc_ldcolortempct,norders,/INVERT
     
;  Plot the location of the center of the lines on the arrays
     
     pos = mc_gridplotpos([0.1,0.1,0.95,0.95],[1,1],0.02,/SQUARE)

     ratio = linesumqaplot.nrows/float(linesumqaplot.ncols)

     if ratio ne 1 then begin

        if ratio lt 1 then pos[3] = pos[3]-(pos[3]-pos[1])*ratio
        if ratio gt 1 then pos[2] = pos[2]-(pos[2]-pos[0])/ratio

     endif
                      
     plot,[1],[1],/NODATA,XRANGE=[0,ncols-1],YRANGE=[0,nrows-1],FONT=0, $
          POSITION=pos[*,0],XTITLE='Column (pixels)', $
          YTITLE='Row (pixels)',/XSTY,/YSTY

     for i = 0,norders-1 do begin

        x = findgen(xranges[1,i]-xranges[0,i]+1)+xranges[0,i]        
        topslit = poly(x,edgecoeffs[*,1,i])
        botslit = poly(x,edgecoeffs[*,0,i])
        !p.thick = 1
        oplot,x,botslit
        oplot,x,topslit

        midx = total(xranges[1,i]+xranges[0,i])/2.
        midy = total(poly(midx,edgecoeffs[*,1,i])+ $
                     poly(midx,edgecoeffs[*,0,i]))/2.
        xyouts,midx,midy,strtrim(orders[i]),FONT=0,ALIGNMENT=0.5              
        
     endfor
     !p.thick = 3

     for i = 0,norders-1 do begin
        
        z = where(fix(oo) eq orders[i],cnt)
        if cnt ne 0 then begin
           
           plotsym,0,0.9,/FILL
           oplot,xx[z],yy[z],PSYM=8
           plotsym,0,0.7,/FILL
           oplot,xx[z],yy[z],COLOR=2+i,PSYM=8
           
        endif
        
   endfor

;  Now plot a surface map of the delta Xs

     fitlines2dxd_plotarraypoints,ncols,nrows,xx,yy,deltax, $
                                  'Line !9D!Xx','!9D!Xx (pixels)', $
                                  orders,edgecoeffs,xranges
     
;  Plot the marginalized delta Xs of each line (how wide each line is in x)

     fitlines2dxdxd_plotmarginals,ncols,nrows,xx,yy,deltax, $
                                  'Line !9D!Xx (pixels)', $
                                  orders,oo

;  Now plot a surface map of the delta Ys

     fitlines2dxd_plotarraypoints,ncols,nrows,xx,yy,deltay, $
                                  'Line !9D!Xy','!9D!Xy (pixels)', $
                                  orders,edgecoeffs,xranges
     
;  Plot the marginalized delta Ys of each line (how tall each line is in x)

     fitlines2dxdxd_plotmarginals,ncols,nrows,xx,yy,deltay, $
                                  'Line !9D!Xy (pixels)', $
                                  orders,oo

;  Now plot a surface map of the slopes

     fitlines2dxd_plotarraypoints,ncols,nrows,xx,yy,slope, $
                                  'Line !9D!Xx/!9D!Xy', $
                                  '!9D!Xx/!9D!Xy (pixels)', $
                                  orders,edgecoeffs,xranges
     
;  Plot the marginalized slopes of each line 

     fitlines2dxdxd_plotmarginals,ncols,nrows,xx,yy,slope, $
                                  'Line !9D!Xx/!9D!Xy (pixels)', $
                                  orders,oo

;  Now plot a surface map of the angles

     fitlines2dxd_plotarraypoints,ncols,nrows,xx,yy,angle, $
                                  'Line Angles', $
                                  'Angle (degrees)', $
                                  orders,edgecoeffs,xranges
     
;  Plot the marginalized slopes of each line 

     fitlines2dxdxd_plotmarginals,ncols,nrows,xx,yy,angle, $
                                  'Angle (degrees)', $
                                  orders,oo
     
     mc_setpsdev,/CLOSE,/CONVERT,/ERASE
     
  endif

return, lstruc
  
end
