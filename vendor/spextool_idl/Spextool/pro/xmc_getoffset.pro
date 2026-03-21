;+
; NAME:
;     xmc_getoffset
;    
; PURPOSE:
;     Determines the offset between spectra and the atmospheric transmission
;
; CATEGORY:
;     Spectroscopy
;   
; CALLING SEQUENCE:
;     xmc_getoffset,wave,stack,files,offsets,wrange,XTITLE=xtitle,$
;                   YTITLE=ytitle,GROUP_LEADER=group_leader,CANCEL=cancel
;
; INPUTS:
;     wave  - A 1-D wavelength array
;     stack - An array [*,nspec] of spectra
;     files - A string array giving the file names.
;        
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     GROUP_LEADER - The widget ID of an existing widget that serves
;                    as "group leader" for the newly-created widget. 
;     XTITLE       - A string giving the x-axis title
;     YTITLE       - A string giving the y-axis title
;     CANCEL       - Set on return if there is a problem
;     
; OUTPUTS:
;     offsets - An array [nspec] of pixel offsets
;     wrange  - A 2 element array given the wavelength
;               range over which the offset were determined.
;     
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     xmc_getoffset_state
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     Select the region to scale by typing 's' and choose with the
;     button on top.
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     2019-03-01 - Written by M. Cushing, University of Toledo
;-
;
;==============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;==============================================================================
;
pro xmc_getoffset_initcommon,wave,stack,files,wtrans,ttrans,XTITLE=xtitle,$
                             YTITLE=ytitle

  common xmc_getoffset_state, state
  
  cleanplot,/SILENT
  
  nspec = (size(stack))[2]
  mc_medcomb,stack,med,CANCEL=cancel
  if cancel then return
  
  yrange = [0.4*min(med,/NAN,MAX=max),1.2*max]
  wrange = (max(wave,MIN=min,/NAN)-min)
  
  if n_elements(XTITLE) eq 0 then xtitle = ''
  if n_elements(YTITLE) eq 0 then ytitle = ''

;  Build three structures which will hold important info.
  
  w = {idwin:0,$
       keyboard:0L,$
       message:0L,$
       plot_base:0L,$
       plotwin:0,$
       xmc_getoffset_base:0L,$
       xmin_fld:[0L,0L],$
       xmax_fld:[0L,0L],$
       ymin_fld:[0L,0L],$
       ymax_fld:[0L,0L]}
  
  r = {allorders:0,$
       colors:[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,$
               1,2,3,4,6,7,8,9,10,11,12,13,14,15,16],$
       lines:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,$
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,$
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,$
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,$
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,$
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,$
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,$
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,$
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,$
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,$
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,$
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,$
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,$
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,$
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,$
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,$
              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,$
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],$
       files:files,$
       
       cancel:0,$
       cursormode:'None',$
       maxspec:0,$
       nspec:nspec,$
       srange:[!values.f_nan,!values.f_nan],$
       offsets:fltarr(nspec)+1.0,$
       div:0.0}
  
  d = {oflux:stack,$
       owave:wave,$
       ttrans:ttrans,$
       wflux:stack,$
       wtrans:wtrans}
  
  p = {buffer:[0.,0.],$
       cursor:0,$
       idwin_wid:0L,$
       pixmap_wid:0L,$
       plotwin_wid:0L,$
       plotxrange:[min(wave,MAX=max),max],$
       plotyrange:yrange,$
       plotabsyrange:yrange,$
       plotabsxrange:[min(wave,MAX=max),max],$
       idratio:5.0,$
       plotsize:[700,550],$
       pscale:!p,$
       xscale:!x,$
       xtitle:xtitle,$
       yscale:!y,$
       ytitle:ytitle,$
       reg:[[!values.f_nan,!values.f_nan],$
            [!values.f_nan,!values.f_nan]]}
  
;  Load the three structures in the state structure.
  
  state = {w:w,r:r,d:d,p:p}
  
end
;
;===============================================================================
;
pro xmc_getoffset_plotid

  common xmc_getoffset_state
  
  wset, state.p.idwin_wid

  polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM
  
  plot,indgen(state.r.nspec+2),/NODATA,XRANGE=[0,1],$
       YRANGE=[0,state.r.nspec+1],XSTY=5,YSTY=5,XMARGIN=[0,0],YMARGIN=[0,0],$
       /NOERASE
  
  for i = 0, state.r.nspec-1 do begin
     
     xyouts,0.1,state.r.nspec-i,strtrim(state.r.files[i],2),$
            COLOR=state.r.colors[i],CHARSIZE=1.5
     
  endfor

end
;
;===============================================================================
;
pro xmc_getoffset_plotspec,LINES=lines

  common xmc_getoffset_state

  plot,[1],[1],/NODATA,XSTY=5,YSTY=5,YRANGE=[0,1], $
       XRANGE=state.p.plotxrange,/NOERASE,BACKGROUND=20,CHARSIZE=1.5
  
  z = where(state.d.wtrans gt state.p.plotxrange[0] and $
            state.d.wtrans lt state.p.plotxrange[1])

  oplot,state.d.wtrans[z],state.d.ttrans[z],COLOR=5,PSYM=10  

  plot, state.d.owave,state.d.wflux[*,0],/NODATA,/XSTY,YSTY=9,$
        YRANGE=state.p.plotyrange,XRANGE=state.p.plotxrange, $
        XTITLE=state.p.xtitle,YTITLE=state.p.ytitle, CHARSIZE=1.5,/NOERASE

  ticks = ['0.0','0.2','0.4','0.6','0.8','1.0']
  axis,YAXIS=1,YTICKS=5,YTICKNAME=ticks,YMINOR=2,COLOR=5, $
       CHARSIZE=1.5
  
  state.p.pscale = !p
  state.p.xscale = !x
  state.p.yscale = !y
  
  for i = 0, state.r.nspec-1 do begin

     oplot,state.d.owave,state.d.wflux[*,i],COLOR=state.r.colors[i], $
           LINESTYLE=state.r.lines[i],PSYM=10
     
  endfor
  
;  Plot scale region lines
  
  if keyword_set(LINES) then begin
          
     z = where(finite(state.r.srange) eq 1,count)
     if count eq 0 then goto, cont
     
     for i = 0, count-1 do plots,[state.r.srange[i],state.r.srange[i]],$
                                 !y.crange,COLOR=7,LINESTYLE=2
     
  endif

cont:

end
;
;===============================================================================
; 
pro xmc_getoffset_plotupdate,LINES=lines

  common xmc_getoffset_state

  wset, state.p.pixmap_wid
  erase
  polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM
  xmc_getoffset_plotspec,LINES=lines
  
  wset, state.p.plotwin_wid
  device, COPY=[0,0,state.p.plotsize[0],state.p.plotsize[1],0,0,$
                state.p.pixmap_wid]
  
end
;
;==============================================================================
;
pro xmc_getoffset_resize,event

  common xmc_getoffset_state

  widget_control, state.w.xmc_getoffset_base, TLB_GET_SIZE=size
  

  state.p.plotsize[0]=(size[0]-state.p.buffer[0])
  state.p.plotsize[1]=size[1]-state.p.buffer[1]

  widget_control, state.w.plotwin,UPDATE=0
  widget_control,state.w.plotwin,DRAW_XSIZE=state.p.plotsize[0]  
  widget_control,state.w.plotwin,DRAW_YSIZE=state.p.plotsize[1]  
  widget_control, state.w.plotwin,UPDATE=1

  wdelete,state.p.pixmap_wid

  window, /FREE, /PIXMAP,XSIZE=state.p.plotsize[0],$
          YSIZE=state.p.plotsize[1]
  state.p.pixmap_wid = !d.window

  widget_control, state.w.idwin,UPDATE=0
  widget_control,state.w.idwin,DRAW_XSIZE=150
  widget_control,state.w.idwin,DRAW_YSIZE=state.p.plotsize[1] 
  widget_control, state.w.idwin,UPDATE=1

  xmc_getoffset_plotupdate,/LINES
  xmc_getoffset_plotid

end
;
;===============================================================================
;
pro xmc_getoffset_getoffset

  common xmc_getoffset_state

  lag = findgen(13)-6
  lag = findgen(21)-10  
  ndat = n_elements(state.d.owave)
  pix = findgen(ndat)
  linterp,state.d.wtrans,state.d.ttrans,state.d.owave,a
  a = a/median(a,/EVEN)

  z = where(state.d.owave gt state.r.srange[0] and $
            state.d.owave lt state.r.srange[1], npixels)
    
  for i = 0,state.r.nspec-1 do begin

     d = (state.d.oflux)[*,i]
     d = d/median(d,/EVEN)
          
     corr = c_correlate(d[z],a[z],lag)
     
     max = max(corr,idx)
     offset = lag[idx]
     
;     coeff = mc_polyfit1d(lag,corr,2,/SILENT)
;     linterp,mc_poly1dderiv(lag,coeff),lag,0,offset     
;     print, offset

;     corr =  corr-min(corr)     
;     offset = total(lag*corr)/total(corr)

;     print, offset
;
;     window, 2
;     plot,lag,corr
;     plots,[offset,offset],!y.crange,COLOR=2
;;     re = ' '
;     read, re

     linterp,pix,(state.d.oflux)[*,i],pix-total(offset),nd
     
     state.d.wflux[*,i] = nd
     state.r.offsets[i] = offset
     
  endfor

end
;
;===============================================================================
;
pro xmc_getoffset_setminmax

  common xmc_getoffset_state

  widget_control, state.w.xmin_fld[1],SET_VALUE=strtrim(state.p.plotxrange[0],2)
  widget_control, state.w.xmax_fld[1],SET_VALUE=strtrim(state.p.plotxrange[1],2)
  widget_control, state.w.ymin_fld[1],SET_VALUE=strtrim(state.p.plotyrange[0],2)
  widget_control, state.w.ymax_fld[1],SET_VALUE=strtrim(state.p.plotyrange[1],2)
  
end
;
;===============================================================================
;
pro xmc_getoffset_zoom,IN=in,OUT=out

  common xmc_getoffset_state

  delabsx = state.p.plotabsxrange[1]-state.p.plotabsxrange[0]
  delx    = state.p.plotxrange[1]-state.p.plotxrange[0]
  
  delabsy = state.p.plotabsyrange[1]-state.p.plotabsyrange[0]
  dely    = state.p.plotyrange[1]-state.p.plotyrange[0]
  
  xcen = state.p.plotxrange[0]+delx/2.
  ycen = state.p.plotyrange[0]+dely/2.
  
  case state.r.cursormode of 
     
     'XZoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.p.plotxrange = [xcen-hwin,xcen+hwin]
        xmc_getoffset_plotupdate,/LINES
        
     end
     
     'YZoom': begin
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.p.plotyrange = [ycen-hwin,ycen+hwin]
        xmc_getoffset_plotupdate,/LINES
        
     end
     
     'Zoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.p.plotxrange = [xcen-hwin,xcen+hwin]
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.p.plotyrange = [ycen-hwin,ycen+hwin]
        
        xmc_getoffset_plotupdate,/LINES
        
     end
     
     else:
     
  endcase
  
end
;
;==============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;==============================================================================
;
pro xmc_getoffset_event,event

  common xmc_getoffset_state
  
  widget_control, event.id,  GET_UVALUE = uvalue
  
  case uvalue of
     
     'Accept': widget_control, event.top, /DESTROY
     
     'Cancel': begin
        
        state.r.cancel = 1
        widget_control, event.top, /DESTROY
        
     end
     
     'Keyboard': begin
        
        case strtrim(event.ch,2) of 
           
           'c': begin           ; Clear
              
              state.r.cursormode = 'None'
              state.p.reg = !values.f_nan
              state.r.srange = !values.f_nan
              xmc_getoffset_plotupdate,/LINES
              
           end
           
           'i': xmc_getoffset_zoom,/IN
           
           'o': xmc_getoffset_zoom,/OUT
           
           's': begin
              
              state.r.cursormode = 'Select'
              state.p.reg = !values.f_nan
              state.r.srange = !values.f_nan
              state.d.wflux = state.d.oflux
              xmc_getoffset_plotupdate
              
           end
           
           'w': begin
              
              state.p.plotxrange = state.p.plotabsxrange
              state.p.plotyrange = state.p.plotabsyrange
              xmc_getoffset_plotupdate,/LINES
              xmc_getoffset_setminmax
              
           end
           
           'x': begin 
              
              state.r.cursormode = 'XZoom'
              state.p.reg = !values.f_nan
              
           end
           
           'y': begin 
              
              state.r.cursormode = 'YZoom'
              state.p.reg = !values.f_nan
              
           end
           
           'z': begin
              
              state.r.cursormode = 'Zoom'
              state.p.reg = !values.f_nan
              
           end
           
           else:
           
        endcase
        
     end
     
     'Help Done': widget_control, event.top, /DESTROY
     
     
  endcase
  
cont: 

end
;
;===============================================================================
;
pro xmc_getoffset_minmax_event,event

  common xmc_getoffset_state
  
  xmin = mc_cfld(state.w.xmin_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  xmin2 = mc_crange(xmin,state.p.plotxrange[1],'X Min',/KLT,$
                 WIDGET_ID=state.w.xmc_getoffset_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.w.xmin_fld[0],SET_VALUE=state.p.plotxrange[0]
     return
     
  endif else state.p.plotxrange[0] = xmin2
  
  xmax = mc_cfld(state.w.xmax_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  xmax2 = mc_crange(xmax,state.p.plotxrange[0],'X Max',/KGT,$
                 WIDGET_ID=state.w.xmc_getoffset_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.w.xmax_fld[0],SET_VALUE=state.p.plotxrange[1]
     return
     
  endif else state.p.plotxrange[1] = xmax2
  
  ymin = mc_cfld(state.w.ymin_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  ymin2 = mc_crange(ymin,state.p.plotyrange[1],'Y Min',/KLT,$
                 WIDGET_ID=state.w.xmc_getoffset_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.w.ymin_fld[0],SET_VALUE=state.p.plotyrange[0]
     return
     
  endif else state.p.plotyrange[0] = ymin2
  
  ymax = mc_cfld(state.w.ymax_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  ymax2 = mc_crange(ymax,state.p.plotyrange[0],'Y Max',/KGT,$
                 WIDGET_ID=state.w.xmc_getoffset_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.w.ymax_fld[0],SET_VALUE=state.p.plotyrange[1]
     return
     
  endif else state.p.plotyrange[1] = ymax2
  
  xmc_getoffset_plotupdate,/LINES
  
end
;
;******************************************************************************
;
pro xmc_getoffset_winevent,event

common xmc_getoffset_state

widget_control, event.id,  GET_UVALUE = uvalue

;  Check to see if it is a TRACKING event.

if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin

    if event.enter eq 0 then widget_control, state.w.keyboard, SENSITIVE=0
    wset, state.p.plotwin_wid
    device, COPY=[0,0,state.p.plotsize[0],state.p.plotsize[1],0,0,$
                  state.p.pixmap_wid]

    goto, cont
    
endif

;  If not, set the keyboard focus and active window.

widget_control, state.w.keyboard, /INPUT_FOCUS, /SENSITIVE
wset, state.p.plotwin_wid

!p = state.p.pscale
!x = state.p.xscale
!y = state.p.yscale
x  = event.x/float(state.p.plotsize[0])
y  = event.y/float(state.p.plotsize[1])
xy = convert_coord(x,y,/NORMAL,/TO_DATA)

if event.type eq 1 then begin

    case state.r.cursormode of 

        'Select': begin

            z = where(finite(state.p.reg) eq 1,count)
            if count eq 0 then begin
                
                state.p.reg[*,0] = xy[0:1]
                wset, state.p.pixmap_wid
                plots, [event.x,event.x],[0,state.p.plotsize[1]],COLOR=7,$
                  /DEVICE,LINESTYLE=2
                wset, state.p.plotwin_wid
                device, COPY=[0,0,state.p.plotsize[0],state.p.plotsize[1],0,$
                              0,state.p.pixmap_wid]


            endif else begin 
                
                state.p.reg[*,1] = xy[0:1]
                tmp = reform(state.p.reg[0,*])
                tmp = tmp[sort(tmp)]
                result = mc_crange(tmp,state.p.plotabsxrange,$
                                'Scale Wavelength Range',/KGT,/KLT,$
                                WIDGET_ID=state.w.xmc_getoffset_base,$
                                CANCEL=cancel)
                if not cancel then state.r.srange = tmp

                xmc_getoffset_getoffset
                xmc_getoffset_plotupdate,/LINES
                state.r.cursormode = 'None'
                state.p.reg = !values.f_nan
                
            endelse
    
        end

        'XZoom': begin

            z = where(finite(state.p.reg) eq 1,count)
            if count eq 0 then begin
                
                state.p.reg[*,0] = xy[0:1]
                wset, state.p.pixmap_wid
                plots, [event.x,event.x],[0,state.p.plotsize[1]],COLOR=2,$
                  /DEVICE,LINESTYLE=2
                wset, state.p.plotwin_wid
                device, COPY=[0,0,state.p.plotsize[0],state.p.plotsize[1],0,$
                              0,state.p.pixmap_wid]

            endif else begin

                state.p.reg[*,1] = xy[0:1]
                state.p.plotxrange = [min(state.p.reg[0,*],max=m),m]
                state.r.cursormode = 'None'
                state.p.reg = !values.f_nan
                xmc_getoffset_plotupdate,/LINES
                xmc_getoffset_setminmax
                
            endelse

        end

        'YZoom': begin

            z = where(finite(state.p.reg) eq 1,count)
            if count eq 0 then begin
                
                state.p.reg[*,0] = xy[0:1]
                wset, state.p.pixmap_wid
                plots, [0,state.p.plotsize[0]],[event.y,event.y],COLOR=2,$
                  /DEVICE,LINESTYLE=2

                wset, state.p.plotwin_wid
                device, COPY=[0,0,state.p.plotsize[0],state.p.plotsize[1],0,$
                              0,state.p.pixmap_wid]
               
            endif else begin

                state.p.reg[*,1] = xy[0:1]
                state.p.plotyrange = [min(state.p.reg[1,*],max=m),m]
                state.r.cursormode = 'None'
                state.p.reg = !values.f_nan
                xmc_getoffset_plotupdate,/LINES
                xmc_getoffset_setminmax
                
            endelse

        end
        
        'Zoom': begin
            
            z = where(finite(state.p.reg) eq 1,count)
            if count eq 0 then state.p.reg[*,0] = xy[0:1] else begin 
                
                state.p.reg[*,1] = xy[0:1]
                state.p.plotxrange   = [min(state.p.reg[0,*],MAX=max),max]
                state.p.plotyrange   = [min(state.p.reg[1,*],MAX=max),max]
                xmc_getoffset_plotupdate,/LINES
                xmc_getoffset_setminmax
                state.r.cursormode   = 'None'
                state.p.reg = !values.f_nan
                
            endelse
            
        end

        else:

    endcase

endif

;  Copy the pixmaps and draw the lines.

wset, state.p.plotwin_wid
device, copy=[0,0,state.p.plotsize[0],state.p.plotsize[1],0,0,$
              state.p.pixmap_wid]

case state.r.cursormode of 

    'XZoom': plots, [event.x,event.x],[0,state.p.plotsize[1]],COLOR=2,/DEVICE

    'YZoom': plots, [0,state.p.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE

    'Zoom': begin

        plots, [event.x,event.x],[0,state.p.plotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.p.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA)
        plots,[state.p.reg[0,0],state.p.reg[0,0]],[state.p.reg[1,0],xy[1]],$
          LINESTYLE=2,COLOR=2
        plots, [state.p.reg[0,0],xy[0]],[state.p.reg[1,0],state.p.reg[1,0]],$
          LINESTYLE=2,COLOR=2
        
    end

    else: begin

        plots, [event.x,event.x],[0,state.p.plotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.p.plotsize[0]],[event.y,event.y],COLOR=2,/DEVICE

    end

endcase

;  Update cursor tracker

label = 'Cursor X: '+strtrim(xy[0],2)+', Y:'+strtrim(xy[1],2)
widget_control,state.w.message,SET_VALUE=label
    
cont:
   
end
;
;===============================================================================
;
; ------------------------------Main Program-------------------------------- 
;
;===============================================================================
;
pro xmc_getoffset,wave,stack,files,wtrans,ttrans,sstack,offsets,wrange, $
                  XTITLE=xtitle,YTITLE=ytitle,GROUP_LEADER=group_leader, $
                  CANCEL=cancel
  
  if n_params() lt 3 then begin
    
     print, 'Syntax - xmc_getoffset,wave,stack,files,offsets,wrange,$'
     print, '                       XTITLE=xtitle,YTITLE=ytitle,$'
     print, '                       GROUP_LEADER=group_leader,CANCEL=cancel'
    cancel = 1
    return
    
 endif 

  cancel = mc_cpar('xmc_getoffset',wave,1,'Wave',[2,3,4,5],1)
  if cancel then return
  cancel = mc_cpar('xmc_getoffset',stack,2,'Stack',[2,3,4,5],[1,2])
  if cancel then return

  xmc_getoffset_initcommon,wave,stack,files,wtrans,ttrans,XTITLE=xtitle, $
                           YTITLE=ytitle
  
  if n_elements(GROUP_LEADER) ne 0 then widget_control, group_leader, $
     SENSITIVE=0
  
  common xmc_getoffset_state
  
;  Build the widget.

  mc_getfonts,buttonfont,textfont
  
  state.w.xmc_getoffset_base = widget_base(TITLE='xmc_getoffset', $
                                           GROUP_LEADER=group_leader,$
                                           /TLB_SIZE_EVENTS,$
                                           /COLUMN)

     button = widget_button(state.w.xmc_getoffset_base,$
                            FONT=buttonfont,$
                            EVENT_PRO='xmc_getoffset_event',$
                            VALUE='Cancel',$
                            UVALUE='Cancel')
     
        col1_base = widget_base(state.w.xmc_getoffset_base,$
                                EVENT_PRO='xmc_getoffset_event',$
                                FRAME=2,$
                                /COLUMN)
        
           state.w.message = widget_text(col1_base, $
                                         YSIZE=1)

           state.w.keyboard = widget_text(col1_base, $
                                          /ALL_EVENTS,$
                                          SCR_XSIZE=1, $
                                          SCR_YSIZE=1, $
                                          EVENT_PRO='xmc_getoffset_event',$
                                          UVALUE='Keyboard',$
                                          VALUE='')
         
           state.w.plot_base = widget_base(col1_base,$
                                           /ROW)

              state.w.idwin = widget_draw(state.w.plot_base,$
                                          XSIZE=150,$
                                          YSIZE=state.p.plotsize[1],$
                                          UVALUE='ID Window')
              
            
              state.w.plotwin = widget_draw(state.w.plot_base,$
                                            XSIZE=state.p.plotsize[0],$
                                            YSIZE=state.p.plotsize[1],$
                                            /TRACKING_EVENTS,$
                                            /BUTTON_EVENTS,$
                                            /MOTION_EVENTS,$
                                            EVENT_PRO='xmc_getoffset_winevent',$
                                            UVALUE='Plot Window 1')
              
           row_base = widget_base(col1_base,$
                                  /ROW)
         
              xmin = coyote_field2(row_base,$
                                   LABELFONT=buttonfont,$
                                   FIELDFONT=textfont,$
                                   TITLE='X Min:',$
                                   UVALUE='X Min',$
                                   XSIZE=12,$
                                   EVENT_PRO='xmc_getoffset_minmax_event',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.w.xmin_fld = [xmin,textid]
              
              xmax = coyote_field2(row_base,$
                                   LABELFONT=buttonfont,$
                                   FIELDFONT=textfont,$
                                   TITLE='X Max:',$
                                   UVALUE='X Max',$
                                   XSIZE=12,$
                                   EVENT_PRO='xmc_getoffset_minmax_event',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.w.xmax_fld = [xmax,textid]
              
              ymin = coyote_field2(row_base,$
                                   LABELFONT=buttonfont,$
                                   FIELDFONT=textfont,$
                                   TITLE='Y Min:',$
                                   UVALUE='Y Min',$
                                   XSIZE=12,$
                                   EVENT_PRO='xmc_getoffset_minmax_event',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.w.ymin_fld = [ymin,textid]
              
              ymax = coyote_field2(row_base,$
                                   LABELFONT=buttonfont,$
                                   FIELDFONT=textfont,$
                                   TITLE='Y Max:',$
                                   UVALUE='Y Max',$
                                   XSIZE=12,$
                                   EVENT_PRO='xmc_getoffset_minmax_event',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
              state.w.ymax_fld = [ymax,textid]
              
        accept = widget_button(state.w.xmc_getoffset_base,$
                               EVENT_PRO='xmc_getoffset_event',$
                               FONT=buttonfont,$
                               VALUE='Accept',$
                               UVALUE='Accept')            
             
; Get things running.  Center the widget using the Fanning routine.
          
  cgcentertlb,state.w.xmc_getoffset_base
  widget_control, state.w.xmc_getoffset_base, /REALIZE
        
;  Get plotwin ids
          
  widget_control, state.w.plotwin, GET_VALUE = x
  state.p.plotwin_wid = x
  
  widget_control, state.w.idwin, GET_VALUE = x
  state.p.idwin_wid = x
  
  window, /FREE, /PIXMAP,XSIZE=state.p.plotsize[0],$
          YSIZE=state.p.plotsize[1]
  state.p.pixmap_wid = !d.window
  polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM
  
  mc_mkct
  xmc_getoffset_plotid
  xmc_getoffset_plotupdate,/LINES
  xmc_getoffset_setminmax
  
;  Get sizes of things now for resizing

  widget_control, state.w.xmc_getoffset_base, TLB_GET_SIZE=result
  
  widget_geom = widget_info(state.w.xmc_getoffset_base, /GEOMETRY)
  
  state.p.buffer[0] = widget_geom.xsize-state.p.plotsize[0]
  state.p.buffer[1] = widget_geom.ysize-state.p.plotsize[1]
  
; Start the Event Loop. This will be a blocking program.
  
  XManager, 'xmc_getoffset', $
            EVENT_HANDLER='xmc_getoffset_resize',$
            state.w.xmc_getoffset_base
  
  cancel = state.r.cancel
  
  if cancel then begin
     
  endif else begin

     sstack = state.d.wflux
     offsets = state.r.offsets
     wrange = state.r.srange
     
  endelse
  
  if n_elements(GROUP_LEADER) ne 0 then widget_control, group_leader, $
     SENSITIVE=1
  
end
