;+
; NAME:
;     xzoomplot
;
; PURPOSE:
;     General purpose plotting widget.
;
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xzoomplot,x,y,PSYM=psym,XRANGE=xrange,YRANGE=yrange,XLOG=xlog, $
;               YLOG=ylog,YTITLE=ytitle,XTITLE=xtitle,TITLE=title, $
;               POSITION=position,PLOTWINSIZE=plotwinsize,CANCEL=cancel
; INPUTS:
;     x - The independent array
;     y - The dependent array
;    
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     PSYM        - Standard IDL PSYM keyword.
;     XRANGE      - The initial xrange of the plot
;     YRANGE      - The initial yrange of the plot
;     XLOG        - Set to plot the xaxis logarithmically 
;     YLOG        - Set to plot the xaxis logarithmically 
;     XTITLE      - The xtitle
;     YTITLE      - The ytitle
;     TITLE       - The title
;     POSITION    - The normalized coordinates of the location of the
;                   widget.  [0,0] = top left, [1,1] = bottom right.
;     PLOTWINSIZE - The size of the plot windwo in normalized
;                  widget.  [0,0] = top left, [1,1] = bottom right.
;     CANCEL      - Set on return if there is a problem
;     
; OUTPUTS:
;     None
;     
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     xzoomplot_state
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     Type 'h' or '?' for the help file
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     2002       - Written by M. Cushing, Institute for Astronomy, UH
;     2003-06-11 - Added XRANGE and YRANGE keywords
;     2003-06-11 - Added 'a' cursor mode
;     2004-02-04 - Added XLOG and YLOG keywords
;     2008-06-10 - Added lines and labels inputs.
;     2009-04-21 - Added the WPOS keyword.
;     2014-06-19 - Changed WPOS to POSITION.
;     2014-08-05 - Added PSM keyword.
;     2016-10-06 - Complete rewrite, added slider, removed control panel.
;-
;
;===============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;===============================================================================
;
pro xzoomplot_startup,position,plot_size

  common xzoomplot_state, state
  
  cleanplot,/SILENT

;  Get screen size
  
  screensize = get_screen_size()
  
;  Load the fonts
  
  mc_getfonts,buttonfont,textfont

  state = {absxrange:[0.,0.],$
           absyrange:[0.,0.],$
           buffer:[0L,0L],$
           but_xlog:0L,$
           but_ylog:0L,$
           buttonfont:buttonfont,$           
           charsize:1.5,$
           color:3,$
           cursor:0L,$
           cursormode:'None',$
           devxy:[0L,0L],$
           message:0L,$
           oflux:ptr_new(2),$
           owave:ptr_new(2),$
           pixmap_wid:0L,$
           plot_size:plot_size*screensize,$
           plotwin:0L,$
           plotwin_wid:0L,$
           pscale:!p,$
           psym:10,$
           reg:[[!values.d_nan,!values.d_nan],$
                [!values.d_nan,!values.d_nan]],$
           slider:0L,$
           sliderval:50,$
           textfont:textfont,$
           thick:1,$
           title:'',$
           xlog:0L,$
           xmax_fld:[0L,0L],$
           xmin_fld:[0L,0L],$
           xrange:[0.,0.],$
           xscale:!x,$
           xtitle:'',$
           xzoomplot_base:0L,$
           ylog:0L,$
           ymax_fld:[0L,0L],$
           ymin_fld:[0L,0L],$
           yrange:[0.,0.],$
           yscale:!x,$
           ytitle:'',$
           zeroline:1L}

;  Build the widget.
  
  state.xzoomplot_base = widget_base(TITLE='Xzoomplot', $
                                     /COLUMN,$
                                     /TLB_SIZE_EVENTS)

     row = widget_base(state.xzoomplot_base,$
                       /ROW,$
                       /TOOLBAR,$
                       /NONEXCLUSIVE,$
                       FRAME=2)
     
        state.but_xlog = widget_button(row,$
                                       VALUE='X Log',$
                                       EVENT_PRO='xzoomplot_event',$
                                       UVALUE='X Log Button',$
                                       FONT=buttonfont)

        state.but_ylog = widget_button(row,$
                                       VALUE='Y Log',$
                                       EVENT_PRO='xzoomplot_event',$
                                       UVALUE='Y Log Button',$
                                       FONT=buttonfont)

        button = widget_button(row,$
                               FONT=buttonfont,$
                               EVENT_PRO='xzoomplot_event',$
                               VALUE='Quit',$
                               UVALUE='Done')
    
     state.message = widget_text(state.xzoomplot_base, $
                                 YSIZE=1)

        
     state.plotwin = widget_draw(state.xzoomplot_base,$
                                 XSIZE=state.plot_size[0],$
                                 YSIZE=state.plot_size[1],$
                                 /TRACKING_EVENTS,$
                                 /BUTTON_EVENTS,$
                                 /MOTION_EVENTS,$
                                 /KEYBOARD_EVENTS,$
                                 EVENT_PRO='xzoomplot_plotwinevent',$
                                 UVALUE='Plot Window')

     row = widget_base(state.xzoomplot_base,$
                       /ROW,$
                       /BASE_ALIGN_LEFT,$
                       FRAME=2)
     
        xmin = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='X Min:',$
                             UVALUE='X Min',$
                             XSIZE=12,$
                             EVENT_PRO='xzoomplot_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmin_fld = [xmin,textid]
                
        xmax = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='X Max:',$
                             UVALUE='X Max',$
                             XSIZE=12,$
                             EVENT_PRO='xzoomplot_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmax_fld = [xmax,textid]
        
        ymin = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='Y Min:',$
                             UVALUE='Y Min',$
                             XSIZE=12,$
                             EVENT_PRO='xzoomplot_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymin_fld = [ymin,textid]
        
        ymax = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='Y Max:',$
                             UVALUE='Y Max',$
                             XSIZE=12,$
                             EVENT_PRO='xzoomplot_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymax_fld = [ymax,textid]
     
     state.slider = widget_slider(state.xzoomplot_base,$
                                  UVALUE='Slider',$
                                  EVENT_PRO='xzoomplot_event',$
                                  /DRAG,$
                                  /SUPPRESS_VALUE,$
                                  FONT=buttonfont)
     widget_control, state.slider, SET_VALUE=state.sliderval

; Get things running.  Center the widget using the Fanning routine.
           
  cgcentertlb,state.xzoomplot_base,position[0],position[1]
  widget_control, state.xzoomplot_base, /REALIZE
           
;  Get plotwin ids
           
  widget_control, state.plotwin, GET_VALUE=x
  state.plotwin_wid = x
  
  window, /FREE, /PIXMAP,XSIZE=state.plot_size[0],YSIZE=state.plot_size[1]
  state.pixmap_wid = !d.window
  
;  Get sizes for things.
           
  widget_geom = widget_info(state.xzoomplot_base, /GEOMETRY)
  
  state.buffer[0]=widget_geom.xsize-state.plot_size[0]
  state.buffer[1]=widget_geom.ysize-state.plot_size[1]
  
; Start the Event Loop. This will be a non-blocking program.
  
  XManager, 'xzoomplot', $
            state.xzoomplot_base, $
            EVENT_HANDLER='xzoomplot_resizeevent',$
            /NO_BLOCK
     

end
;
;===============================================================================
;
pro xzoomplot_cleanup,xzoomplot_base

common xzoomplot_state

if n_elements(state) ne 0 then begin

    ptr_free, state.oflux
    ptr_free, state.owave
        
endif
state = 0B

end
;
;******************************************************************************
;
pro xzoomplot_help

  common xzoomplot_state
  
  h = [['Xzoomplot is a fully resizing widget.'],$
       [' '],$
       ['Arrows keys can be used to move the plot window left or right.'],$
       ['Shift+arrows keys can be used to move cross hair left/right or up/down .'],$
       [' '],$
       ['Keyboard commands:'],$
       [' '],$
       ["a - Sets the 'a'bsolute range to the current x and y range"],$
       [' '],$
       ['c - Clear mouse mode.'],$
       ['    Use to clear a zoom, fix, or remove session.'],$
       [' '],$
       ['i - To zoom IN in whatever zoom mode the cursor is currently'],$
       ['    in.'],$
       [' '],$
       ['h - To lauch the help window.'],$
       [' '],$
       ["r - To 'r'eport the current cursor position at the terminal."],$
       [' '],$
       ['o - To zoom OUT in whatever zoom mode the cursor is currently'],$
       ['    in.'],$
       [' '],$
       ['q - To quit..'],$
       [' '],$
       ['w - To plot the entire spectrum'],$
       [' '],$
       ['x - Enters x zoom mode'],$
       ['    Press left mouse button at lower x value and then at upper'],$
       ['    x value.'],$
       [' '],$
       ['y - Enters y zoom mode'],$
       ['    Press left mouse button at lower y value and then at upper'],$
       ['    y value.'],$
       [' '],$
       ['z - Enters zoom mode'],$
       ['    Press the left mouse button in one corner of the zoom box '],$
       ['    and then move the cursor to the other corner and press the '],$
       ['    the left mouse button.'],$
       [' ']]
  
  xmc_displaytext,h,TITLE='Xzoomplot Help File', $
                  GROUP_LEADER=state.xzoomplot_base
  
end

;
;===============================================================================
;
pro xzoomplot_plotspec

  common xzoomplot_state

  !p.multi = 0
  
  color=state.color
  wset, state.pixmap_wid

  polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM

  plot,*state.owave,*state.oflux,/XSTY,/YSTY,YRANGE=state.yrange,$
       XRANGE=state.xrange,/NODATA,CHARTHICK=state.thick,$
       THICK=state.thick,PSYM=state.psym,XTITLE=state.xtitle, $
       YTITLE=state.ytitle,TITLE=state.title, $
       CHARSIZE=state.charsize,XTHICK=state.thick,YTHICK=state.thick, $
       XLOG=state.xlog, YLOG=state.ylog,/NOERASE
  
  oplot, *state.owave,*state.oflux,COLOR=color,THICK=state.thick, $
         PSYM=state.psym
  
  if state.yrange[0] lt 0 and state.yrange[1] gt 0 and $
     ~state.ylog and state.zeroline then plots,!x.crange,[0,0],LINESTYLE=1
  
  
  wset, state.plotwin_wid
  device, copy=[0,0,state.plot_size[0],state.plot_size[1],0,0,state.pixmap_wid]
    
  state.xscale = !x
  state.yscale = !y
  state.pscale = !p
  state.cursor = 1
  
cont:

end
;
;===============================================================================
;
pro xzoomplot_setminmax

  common xzoomplot_state
  
  widget_control, state.xmin_fld[1],SET_VALUE=strtrim(state.xrange[0],2)
  widget_control, state.xmax_fld[1],SET_VALUE=strtrim(state.xrange[1],2)
  widget_control, state.ymin_fld[1],SET_VALUE=strtrim(state.yrange[0],2)
  widget_control, state.ymax_fld[1],SET_VALUE=strtrim(state.yrange[1],2)
  
  xzoomplot_setslider

end
;
;=============================================================================
;
pro xzoomplot_setslider

  common xzoomplot_state

;  Get new slider value
  
  del = state.absxrange[1]-state.absxrange[0]
  midwave = (state.xrange[1]+state.xrange[0])/2.
  state.sliderval = round((midwave-state.absxrange[0])/del*100)
  
  widget_control, state.slider, SET_VALUE=state.sliderval
  
end
;
;******************************************************************************
;
pro xzoomplot_zoom,IN=in,OUT=out

  common xzoomplot_state
  
  delabsx = state.absxrange[1]-state.absxrange[0]
  delx    = state.xrange[1]-state.xrange[0]
  
  delabsy = state.absyrange[1]-state.absyrange[0]
  dely    = state.yrange[1]-state.yrange[0]
  
  xcen = state.xrange[0]+delx/2.
  ycen = state.yrange[0]+dely/2.
  
  case state.cursormode of 
     
     'XZoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.xrange = [xcen-hwin,xcen+hwin]
        
     end
     
     'YZoom': begin
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.yrange = [ycen-hwin,ycen+hwin]
        
     end
     
     'Zoom': begin
        
        z = alog10(delabsx/delx)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsx/2.^z/2.
        state.xrange = [xcen-hwin,xcen+hwin]
        
        z = alog10(delabsy/dely)/alog10(2)
        if keyword_set(IN) then z = z+1 else z=z-1
        hwin = delabsy/2.^z/2.
        state.yrange = [ycen-hwin,ycen+hwin]
        
     end
     
     else:
     
  endcase
  xzoomplot_setminmax
  xzoomplot_plotspec
               
  
end
;
;===============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;===============================================================================
;
pro xzoomplot_event, event

  common xzoomplot_state
  
  widget_control, event.id,  GET_UVALUE = uvalue

  case uvalue of
     
     'Axis Type': begin
        
        if event.value eq 'Xlog' then state.xlog = event.select
        if event.value eq 'Ylog' then state.ylog = event.select
        xzoomplot_plotspec
        
     end
     
     'CharSize': begin
        
        val = mc_cfld(state.charsize_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        state.charsize=val
        xzoomplot_plotspec
        
     end
     
     'Keyboard': begin
        
     end
     
     'Done': widget_control, event.top, /DESTROY
     
     'Slider': begin
        
        del = state.absxrange[1]-state.absxrange[0]
        oldcen = (state.xrange[1]+state.xrange[0])/2.
        newcen = state.absxrange[0]+del*(event.value/100.)
        
        
        state.xrange = state.xrange + (newcen-oldcen)
        xzoomplot_plotspec
        
     end
     
     'Spectrum Color': begin
        
        state.color = event.index+1
        xzoomplot_plotspec
        
     end
     
     'Thick': begin
        
        label = mc_cfld(state.thick_fld,7,/EMPTY,CANCEL=cancel)
        if cancel then return
        state.thick = label
        xzoomplot_plotspec
        
     end
     
     'Title': begin

        label = mc_cfld(state.title_fld,7,/EMPTY,CANCEL=cancel)
        if cancel then return
        state.title = label
        xzoomplot_plotspec
        
     end
     
     'X Log Button': begin
        
        state.xlog = event.select
        xzoomplot_plotspec
        
     end
     
     'X Title': begin
        
        label = mc_cfld(state.xtitle_fld,7,/EMPTY,CANCEL=cancel)
        if cancel then return
        state.xtitle = label
        xzoomplot_plotspec
        
     end
     
     'Y Log Button': begin
        
        state.ylog = event.select
        xzoomplot_plotspec
        
     end
     
     'Y Title': begin
        
        label = mc_cfld(state.ytitle_fld,7,/EMPTY,CANCEL=cancel)
        if cancel then return
        state.ytitle = label
        xzoomplot_plotspec
        
     end
     
     'Zero Line': begin
        
        state.zeroline=event.select
        xzoomplot_plotspec
        
     end
     
  endcase
  
cont: 
  
end
;
;===============================================================================
;
pro xzoomplot_minmaxevent,event

  common xzoomplot_state
  
  xmin = mc_cfld(state.xmin_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  xmin2 = mc_crange(xmin,state.xrange[1],'X Min',/KLT,$
                    WIDGET_ID=state.xzoomplot_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.xmin_fld[0],SET_VALUE=state.xrange[0]
     return

  endif else state.xrange[0] = xmin2

  xmax = mc_cfld(state.xmax_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  xmax2 = mc_crange(xmax,state.xrange[0],'X Max',/KGT,$
                    WIDGET_ID=state.xzoomplot_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.xmax_fld[0],SET_VALUE=state.xrange[1]
     return
     
  endif else state.xrange[1] = xmax2
  
  ymin = mc_cfld(state.ymin_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  ymin2 = mc_crange(ymin,state.yrange[1],'Y Min',/KLT,$
                    WIDGET_ID=state.xzoomplot_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.ymin_fld[0],SET_VALUE=state.yrange[0]
     return
     
  endif else state.yrange[0] = ymin2
  
  ymax = mc_cfld(state.ymax_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  ymax2 = mc_crange(ymax,state.yrange[0],'Y Max',/KGT,$
                    WIDGET_ID=state.xzoomplot_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.ymax_fld[0],SET_VALUE=state.yrange[1]
     return
     
  endif else state.yrange[1] = ymax2
  
  xzoomplot_plotspec
  
end
;
;===============================================================================
;
pro xzoomplot_plotwinevent, event

  common xzoomplot_state
  
  widget_control, event.id,  GET_UVALUE = uvalue

;  Check to see if it is a TRACKING event.

  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin
     
     widget_control, state.plotwin, INPUT_FOCUS=event.enter
     wset, state.plotwin_wid
     device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0, $
                   state.pixmap_wid]
     return
     
     
  endif

;  Check for arrow keys for plot motion move
   
  if event.type eq 6 and event.release eq 0 then begin

     if event.modifiers eq 0 then begin

        case event.key of
           
           5: begin

              del = (state.xrange[1]-state.xrange[0])*0.3
              oldcen = (state.xrange[1]+state.xrange[0])/2.
              newcen = oldcen-del
              
              if newcen lt state.absxrange[0] then return
              state.xrange = state.xrange + (newcen-oldcen)
              xzoomplot_setminmax
              xzoomplot_plotspec
              
           end
           
           6: begin

              del = (state.xrange[1]-state.xrange[0])*0.3
              oldcen = (state.xrange[1]+state.xrange[0])/2.
              newcen = oldcen+del
              
              if newcen gt state.absxrange[1] then return
              state.xrange = state.xrange + (newcen-oldcen)
              xzoomplot_setminmax
              xzoomplot_plotspec
              
           end
           
           else:
           
        endcase

     endif

  endif

;  Check for arrow keys for cursor move
  
  if event.type eq 6 and event.modifiers eq 1 then begin
     
     if event.release eq 1 then begin
        
        widget_control, state.plotwin, /INPUT_FOCUS
        tvcrs,state.devxy[0],state.devxy[1],/DEVICE
        return
        
     endif

     tvcrs,/HIDE_CURSOR
     case event.key of
        
        5: state.devxy = [state.devxy[0]-1,state.devxy[1]]

        6: state.devxy = [state.devxy[0]+1,state.devxy[1]]

        7: state.devxy = [state.devxy[0],state.devxy[1]+1]
        
        8: state.devxy = [state.devxy[0],state.devxy[1]-1]        
        
        else:
        
     endcase
     goto, update
     
  endif
  
  if strtrim(event.ch,2) ne '' then begin
  
  ;  Catch for Clemens' bug.

     if (event.type eq 2) then return
     if (event.press eq 1) then return
     
     case strtrim(event.ch,2) of 
        
        '?': xzoomplot_help
        
        'a': begin
           
           state.absxrange = state.xrange
           state.absyrange=state.yrange
           
        end
        
        'c': begin          
           
           state.cursormode = 'None'
           state.reg = !values.f_nan                
           xzoomplot_plotspec
           
        end
        
        'i': xzoomplot_zoom,/IN
        
        'h': xzoomplot_help

        '?': xzoomplot_help   

        'm': begin 

              !p = state.pscale
              !x = state.xscale
              !y = state.yscale
              xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA,/DOUBLE)
              
              print, xy[0:1]
           
        end
        
        'o': xzoomplot_zoom,/OUT
        
        'q': begin

           widget_control, event.top, /DESTROY
           return

        end
        
        'w': begin
           
           state.xrange = state.absxrange
           state.yrange = state.absyrange
           xzoomplot_plotspec
           xzoomplot_setminmax
           
        end
        
        'x': begin 
           
           state.cursormode = 'XZoom'
           state.reg = !values.f_nan
           
        end
        
        'y': begin 
           
           state.cursormode = 'YZoom'
           state.reg = !values.f_nan
           
        end
        
        'z': begin        
           
           state.cursormode = 'Zoom'
           state.reg = !values.f_nan
           
        end
        
        else:
        
     endcase

  endif

  state.devxy = [event.x,event.y]
      
;  Update cross hair and cursor readout
  
  update:
  
  wset, state.plotwin_wid

  !p = state.pscale
  !x = state.xscale
  !y = state.yscale

  xy = convert_coord(state.devxy[0],state.devxy[1],/DEVICE,/TO_DATA,/DOUBLE)
  
  if event.type eq 1 then begin
     
     if state.cursormode eq 'None' then return
     z = where(finite(state.reg) eq 1,count)
     if count eq 0 then begin
        
        wset, state.pixmap_wid
        state.reg[*,0] = xy[0:1]
        case state.cursormode of
           
           'XZoom': plots, [state.devxy[0],state.devxy[0]],[0,state.plot_size[1]], $
                           COLOR=2,/DEVICE,LINESTYLE=2
           
           'YZoom': plots, [0,state.plot_size[0]],[state.devxy[1],state.devxy[1]], $
                           COLOR=2,/DEVICE,LINESTYLE=2
           
           else:
           
        endcase
        wset, state.plotwin_wid
        device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,$
                      state.pixmap_wid]
        
     endif else begin 
        
        state.reg[*,1] = xy[0:1]
        case state.cursormode of 
           
           'XZoom': state.xrange = [min(state.reg[0,*],MAX=max),max]
           
           'YZoom': state.yrange = [min(state.reg[1,*],MAX=max),max]
           
           'Zoom': begin
              
              state.xrange = [min(state.reg[0,*],MAX=max),max]
              state.yrange = [min(state.reg[1,*],MAX=max),max]
              
           end
           
        endcase
        xzoomplot_plotspec
        xzoomplot_setminmax
        state.cursormode='None'
        
     endelse
     
  endif
  
;  Copy the pixmaps and draw the cross hair or zoom lines.
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,$
                state.pixmap_wid]
  
  case state.cursormode of 
     
     'XZoom': plots, [state.devxy[0],state.devxy[0]],[0,state.plot_size[1]],COLOR=2, $
                     /DEVICE
     
     'YZoom': plots, [0,state.plot_size[0]],[state.devxy[1],state.devxy[1]],COLOR=2, $
                     /DEVICE
     
     'Zoom': begin
        
        plots, [state.devxy[0],state.devxy[0]],[0,state.plot_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot_size[0]],[state.devxy[1],state.devxy[1]],COLOR=2,/DEVICE
        xy = convert_coord(state.devxy[0],state.devxy[1],/DEVICE,/TO_DATA,/DOUBLE)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots, [state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
               LINESTYLE=2,COLOR=2
        
     end
     
     else: begin

        plots, [state.devxy[0],state.devxy[0]],[0,state.plot_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot_size[0]],[state.devxy[1],state.devxy[1]],COLOR=2,/DEVICE
        
     end
     
  endcase
  
;  Update cursor position.
  
  if state.cursor then begin

     mc_tabinv, *state.owave,xy[0],idx
     idx = round(idx)
     label = 'Cursor X: '+strtrim(xy[0],2)+', Y:'+strtrim(xy[1],2)
     label = label+'   Spectrum Col:  '+strtrim(idx[0],2)+'   X: '+$
             strtrim( (*state.owave)[idx],2)+$
             ', Y:'+strtrim( (*state.oflux)[idx],2)
     widget_control,state.message,SET_VALUE=label
     
  endif
  
end
;
;******************************************************************************
;
pro xzoomplot_resizeevent, event

  common xzoomplot_state
  
  widget_control, state.xzoomplot_base, TLB_GET_SIZE=size
  
  
  state.plot_size[0]=size[0]-state.buffer[0]
  state.plot_size[1]=size[1]-state.buffer[1]
  
  widget_control, state.plotwin,UPDATE=0
  widget_control, state.plotwin, DRAW_XSIZE=state.plot_size[0]
  widget_control, state.plotwin, DRAW_YSIZE=state.plot_size[1]
  widget_control, state.plotwin,UPDATE=1
  
  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plot_size[0],YSIZE=state.plot_size[1]
  state.pixmap_wid = !d.window
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,$
                state.pixmap_wid]
  
  xzoomplot_plotspec
  
end
;
;===============================================================================
;
; ------------------------------Main Program-------------------------------- 
;
;===============================================================================
;
pro xzoomplot,x,y,PSYM=psym,XRANGE=xrange,YRANGE=yrange,XLOG=xlog, $
              YLOG=ylog,YTITLE=ytitle,XTITLE=xtitle,TITLE=title, $
              POSITION=position,PLOTWINSIZE=plotwinsize,CANCEL=cancel

  cancel = 0

  mc_mkct
  common xzoomplot_state
  
  if n_params() ge 2 then begin
     
     cancel = mc_cpar('xzoomplot',x,1,'X',[2,3,4,5],1)
     if cancel then return
     cancel = mc_cpar('xzoomplot',y,2,'Y',[2,3,4,5],1)
     if cancel then return
     
     if n_elements(POSITION) eq 0 then position = [0.5,0.5]
     if n_elements(PLOTWINSIZE) eq 0 then plotwinsize=[0.5,0.5]
     if not xregistered('xzoomplot') then xzoomplot_startup,position,plotwinsize
     
     state.ytitle = (n_elements(YTITLE) ne 0) ? ytitle:''
     state.xtitle = (n_elements(XTITLE) ne 0) ? xtitle:''
     state.title  = (n_elements(TITLE) ne 0) ? title:''
     state.xlog   = keyword_set(XLOG)
     widget_control, state.but_xlog,SET_BUTTON=keyword_set(XLOG)
     state.ylog   = keyword_set(YLOG)
     widget_control, state.but_ylog,SET_BUTTON=keyword_set(YLOG)
     state.psym   = (n_elements(PSYM) eq 0)? 10:psym
     
;  Strip NaNs (why again?)

     good = where(finite(x) eq 1)
     *state.owave = x[good]
     *state.oflux = y[good]

;  Get plot ranges

     state.absxrange = [min(x,MAX=xmax,/NAN),xmax]
     state.xrange    = (n_elements(XRANGE) ne 0) ? xrange:state.absxrange

     xx = findgen(n_elements(x))
     smooth = mc_robustsg(xx,y,5,3,0.1,CANCEL=cancel)
     if cancel then return
     
     min = min(smooth[*,1],/NAN,MAX=max)
     del = (max-min)*0.1
     
     state.absyrange = [min-del,max+del]
     state.yrange    = (n_elements(YRANGE) ne 0) ? yrange:state.absyrange

;  Get started
     
     xzoomplot_setminmax             
     xzoomplot_plotspec
     
  endif else begin
     
     cancel = 1
     print, 'Syntax - xzoomplot,x,y,PSYM=psym,XRANGE=xrange,$'
     print, '                   YRANGE=yrange,XLOG=xlog,$'
     print, '                   YLOG=ylog,YTITLE=ytitle,XTITLE=xtitle,$'
     print, '                   TITLE=title,POSITION=position,$'
     print, '                   PLOTWINSIZE=plotwinsize,CANCEL=cancel'
     return
     
  endelse
  

  end


  
