;+
; NAME:
;     xmc_corspec
;
; PURPOSE:
;     To (interactively) determine the pixel shift between two spectra
;
; CALLING SEQUENCE:
;     xmc_corspec,xanchor,yanchor,x,y,offset,JUSTFIT=justfit,TITLE=title, $
;                 ANCHROLABEL=anchorlabel,TESTLABEL=testlabel,MAX=max,$
;                 CANCEL=cancel
;
; INPUTS:
;     xanchor - A 1D array giving the pixel values for the anchor
;               spectrum.
;     yanchor - A 1D array giving the intensity of the anchor
;               spectrum.
;     x       - A 1D array giving the pixel values for the spectrum.
;     y       - A 1D array giving the intensity of the spectrum.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     ANCHORLABEL - If given, the anchor spectrum label.
;                   Default is 'Disk spectrum'
;     TESTLABEL   - If given, the y spectrum label.
;                   Default is 'Observed spectrum'
;     JUSTFIT     - If set, then the widget is not launched and the offset
;                   is simply returned.
;     TITLE       - A scalar string giving a title for the plot.
;     MAX         - Set to find the maximum of the cross correlation
;                   instead of fitting it.
;     CANCEL      - Set on return if there is a problem.
;
; OUTPUTS:
;     offset  - The offset between the two spectra such that y=xanchor+guess
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
;     Compute the cross correlation function and fit the results with
;     a Lorentzian to determine the shift.  If ~JUSTFIT, then a GUI is
;     launched that allows the user to interactively select wavelength
;     range over which to the do the cross correlation.
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-10-08 - Written by M. Cushing, University of Toledo
;-
;
;===============================================================================
;
;---------------------------- Support Procedures -------------------------------
;
;===============================================================================
;
pro corspec_cleanup,base

  widget_control, base, GET_UVALUE = state, /NO_COPY
  if n_elements(state) ne 0 then begin

     ptr_free, state.lag
     ptr_free, state.corr
     ptr_free, state.fit
         
  endif 
  state = 0B

end
;
;===============================================================================
;
pro corspec_corspec,state

  z = where(state.xanchor ge state.corrange[0] and $
            state.xanchor le state.corrange[1],npixels)
  
  linterp,state.x,state.y,state.xanchor[z],ynew,MISSING=0
  lag = findgen(npixels)-fix(npixels/2.)

  corr = c_correlate(state.yanchor[z],ynew,lag)

  max = max(corr,idx)

;  Find the inflection point

  dif = 1
  i = 0
  while dif gt 0 and idx+i+1 lt npixels-1 do begin
     
     dif = corr[idx+i]-corr[idx+i+1]
     i = i + 1
     
  endwhile

  win = i*4  
  *state.lag = lag[(idx-win) > 0 :(idx+win) < (npixels-1)]
  *state.corr = corr[(idx-win) > 0 :(idx+win) < (npixels-1)]

  
  if state.max then begin

     *state.offset = lag[idx]
     *state.fit = !values.f_nan
     
  endif else begin
  
     
;  Fit around the peak
     
     estimates = [max,lag[idx],i/2.,0]
     *state.fit = mpfitpeak(*state.lag,*state.corr,a,NTERMS=5,/LORENTZIAN, $
                            ESTIMATES=estimates)     
     *state.offset = a[1]


  endelse
end
;
;===============================================================================
;
pro corspec_plotupdate,state,REPORT=report

  if keyword_set(REPORT) then begin

     wset, state.plotwin2_wid
     polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM
     plot,*state.lag,*state.corr,PSYM=10,/XSTY,/YSTY,/NOERASE, $
          THICK=state.thick,CHARSIZE=state.charsize,CHARTHICK=state.thick,$
          XTHICK=state.thick,YTHICK=state.thick,$
          XTITLE='Lag (pixels)',YTITLE='Relative Intensity', $
          TITLE='Cross Correlation'
          
     if ~state.max then oplot,*state.lag,*state.fit,PSYM=10,COLOR=3
     plots,[*state.offset,*state.offset],!y.crange,COLOR=3

     xy = convert_coord(!x.crange[1],!y.crange[1],/DATA,/TO_DEVICE)

     xyouts,xy[0]-15,xy[1]-30,'Data',ALIGNMENT=1,/DEVICE, $
            CHARSIZE=state.charsize
     
     xyouts,xy[0]-15,xy[1]-50,'Model',ALIGNMENT=1,/DEVICE, $
            CHARSIZE=state.charsize,COLOR=3

     result = 'dx='+string(*state.offset,FORMAT='(f+5.1)')
     
     xyouts,xy[0]-15,xy[1]-70,result,ALIGNMENT=1,/DEVICE, $
            CHARSIZE=state.charsize,COLOR=3
         
  endif
  
  wset, state.pixmap_wid

  polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM

  plot,state.xanchor,state.yanchor,/XSTY,/YSTY,YRANGE=state.yrange,$
       XRANGE=state.xrange,CHARTHICK=state.thick,$
       THICK=state.thick,PSYM=10,XTITLE='Column (pixels)', $
       YTITLE='Relative Intensity',TITLE=state.title, $
       CHARSIZE=state.charsize,XTHICK=state.thick,YTHICK=state.thick,$
       /NOERASE,/NODATA

  
  oplot, state.xanchor,state.yanchor,COLOR=2,THICK=state.thick,PSYM=10
  oplot, state.x,state.y,THICK=state.thick,PSYM=10

  plots,[state.corrange[0],state.corrange[0]],!y.crange,LINESTYLE=2,COLOR=7, $
        THICK=state.thick
  plots,[state.corrange[1],state.corrange[1]],!y.crange,LINESTYLE=2,COLOR=7,$
        THICK=state.thick

;  Write labels

  xy = convert_coord(!x.crange[1],!y.crange[1],/DATA,/TO_DEVICE)

  xyouts,xy[0]-20,xy[1]-30,state.testlabel,ALIGNMENT=1,/DEVICE, $
         CHARSIZE=state.charsize
  xyouts,xy[0]-20,xy[1]-50,state.anchorlabel,ALIGNMENT=1,/DEVICE, $
         CHARSIZE=state.charsize,COLOR=2
  
  state.xscale = !x
  state.yscale = !y
  state.pscale = !p
  
  wset, state.plotwin1_wid
  device, copy=[0,0,state.specplotsize[0],state.specplotsize[1],0,0, $
                state.pixmap_wid]  
  
end
;
;===============================================================================
;
pro corspec_setminmax,state

  widget_control, state.xmin_fld[1],SET_VALUE=strtrim(state.xrange[0],2)
  widget_control, state.xmax_fld[1],SET_VALUE=strtrim(state.xrange[1],2)
  widget_control, state.ymin_fld[1],SET_VALUE=strtrim(state.yrange[0],2)
  widget_control, state.ymax_fld[1],SET_VALUE=strtrim(state.yrange[1],2)
  
end
;
;===============================================================================
;
pro corspec_zoom,state,IN=in,OUT=out

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

  corspec_setminmax,state
  corspec_plotupdate,state


end
;
;===============================================================================
;
; ------------------------------ Event Handlers --------------------------------
;
;===============================================================================
;
pro corspec_event,event

  widget_control, event.top, GET_UVALUE=state, /NO_COPY
  widget_control, event.id,  GET_UVALUE=uvalue
  
  case uvalue of

     'Accept': begin

        *state.cancel = 0
        widget_control, event.top,/DESTROY

     end

     'Cancel': begin

        *state.cancel = 1
        widget_control, event.top,/DESTROY

     end
     

  endcase
  
end
;
;===============================================================================
;
pro corspec_plotwinevent,event

  widget_control, event.top, GET_UVALUE=state, /NO_COPY

  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin
     
     widget_control, state.plotwin1,INPUT_FOCUS=event.enter

     wset, state.plotwin1_wid
     device, COPY=[0,0,state.specplotsize[0],state.specplotsize[1],0,0,$
                   state.pixmap_wid]
     goto, cont
     
  endif

;  Now check for ASCII characters
  
  if event.type eq 5 and event.release eq 1 then begin
         
     case strtrim(event.ch,2) of 

        'a': begin
           
           state.absxrange = state.xrange
           state.absyrange=state.yrange
           
        end
        
        'c': begin          
           
           state.cursormode = 'None'
           state.reg = !values.f_nan
           corspec_plotupdate,state
           
        end

        'i': corspec_zoom,state,/IN

        'o': corspec_zoom,state,/OUT
        
        's': begin

           state.cursormode = 'Select'
           state.reg=!values.f_nan
           state.corrange=!values.f_nan
           corspec_plotupdate,state
           
        end

        'w': begin
           
           state.xrange = state.absxrange
           state.yrange = state.absyrange
           corspec_setminmax,state
           corspec_plotupdate,state
           
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

  wset, state.plotwin1_wid
     
  !p = state.pscale
  !x = state.xscale
  !y = state.yscale
  x  = event.x/float(state.specplotsize[0])
  y  = event.y/float(state.specplotsize[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA,/DOUBLE)
  
  if event.type eq 1 then begin

     if state.cursormode eq 'Select' then begin

        z = where(finite(state.corrange) eq 1,count)
        if count eq 0 then begin

           state.corrange[0] = xy[0]
           corspec_plotupdate,state
           
        endif else begin 
           
           state.corrange[1] = xy[0]
           x1 = xy[0] < state.corrange[0]
           x2 = xy[0] > state.corrange[0]
           
           state.corrange = [x1,x2]
           corspec_corspec,state
           corspec_plotupdate,state,/REPORT
           state.cursormode = 'None'
           state.reg = !values.f_nan
           
        endelse
        
     endif
  
     z = where(finite(state.reg) eq 1,count)
     if count eq 0 then begin
        
        wset, state.pixmap_wid
        state.reg[*,0] = xy[0:1]
        case state.cursormode of
           
           'XZoom': plots, [event.x,event.x],$
                           [0,state.specplotsize[1]],COLOR=2,/DEVICE, $
                           LINESTYLE=2
           
           'YZoom': plots, [0,state.specplotsize[0]],$
                           [event.y,event.y],COLOR=2,/DEVICE,LINESTYLE=2

           'Mask': begin

              plots,[state.reg[0,0],state.reg[0,0]],!y.crange,COLOR=7,THICK=2,$
                    LINESTYLE=2
              
           end

           else:
           
        endcase
        wset, state.plotwin1_wid
        device, COPY=[0,0,state.specplotsize[0], $
                      state.specplotsize[1],0,0,state.pixmap_wid]
        
     endif else begin 
        
        state.reg[*,1] = xy[0:1]
        case state.cursormode of 
           
           'XZoom': state.xrange = [min(state.reg[0,*],MAX=max),max]
           
           'YZoom': state.yrange = [min(state.reg[1,*],MAX=max),max]
           
           'Zoom': begin
              
              state.xrange = [min(state.reg[0,*],MAX=max),max]
              state.yrange = [min(state.reg[1,*],MAX=max),max]
              
           end

           else:
           
        endcase

        corspec_plotupdate,state
        corspec_setminmax,state
        state.cursormode='None'
        
     endelse
     
  endif
    
;  Copy the pixmaps and draw the cross hair or zoom lines.
  
  wset, state.plotwin1_wid
  device, COPY=[0,0,state.specplotsize[0],state.specplotsize[1],0,0,$
                state.pixmap_wid]
  
  case state.cursormode of 
     
     'XZoom': plots, [event.x,event.x],[0,state.specplotsize[1]], $
                     COLOR=2,/DEVICE
     
     'YZoom': plots, [0,state.specplotsize[0]],[event.y,event.y], $
                     COLOR=2,/DEVICE
     
     'Zoom': begin
        
        plots, [event.x,event.x],[0,state.specplotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.specplotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA,/DOUBLE)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots, [state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
               LINESTYLE=2,COLOR=2
        
     end

     else: begin
        
        plots, [event.x,event.x],[0,state.specplotsize[1]],COLOR=2,/DEVICE
        plots, [0,state.specplotsize[0]],[event.y,event.y],COLOR=2,/DEVICE
        
     end
     
  endcase

;  if not state.freeze then begin
     
;     label = 'Cursor (X,Y): '+strtrim(xy[0],2)+', '+strtrim(xy[1],2)
;     widget_control,state.message,SET_VALUE=label
     
;  endif
  
  widget_control, state.plotwin1,/INPUT_FOCUS

  

cont: 
widget_control, event.top, SET_UVALUE=state, /NO_COPY

getout:
  


end
;
;===============================================================================
;
pro corspec_minmaxevent,event

  widget_control, event.top, GET_UVALUE=state, /NO_COPY
  
  
  xmin = mc_cfld(state.xmin_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  xmin2 = mc_crange(xmin,state.xrange[1],'X Min',/KLT,$
                    WIDGET_ID=state.corspec_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.xmin_fld[0],SET_VALUE=state.xrange[0]
     return

  endif else state.xrange[0] = xmin2

  xmax = mc_cfld(state.xmax_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  xmax2 = mc_crange(xmax,state.xrange[0],'X Max',/KGT,$
                    WIDGET_ID=state.corspec_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.xmax_fld[0],SET_VALUE=state.xrange[1]
     return
     
  endif else state.xrange[1] = xmax2
  
  ymin = mc_cfld(state.ymin_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  ymin2 = mc_crange(ymin,state.yrange[1],'Y Min',/KLT,$
                    WIDGET_ID=state.corspec_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.ymin_fld[0],SET_VALUE=state.yrange[0]
     return
     
  endif else state.yrange[0] = ymin2
  
  ymax = mc_cfld(state.ymax_fld,4,/EMPTY,CANCEL=cancel)
  if cancel then return
  ymax2 = mc_crange(ymax,state.yrange[0],'Y Max',/KGT,$
                    WIDGET_ID=state.corspec_base,CANCEL=cancel)
  if cancel then begin
     
     widget_control, state.ymax_fld[0],SET_VALUE=state.yrange[1]
     return
     
  endif else state.yrange[1] = ymax2
  
  corspec_plotupdate,state

  widget_control, state.corspec_base, SET_UVALUE=state, /NO_COPY
  
end
;
;===============================================================================
;
pro corspec_resizeevent,event

  widget_control, event.top, GET_UVALUE=state, /NO_COPY
  
  widget_control, state.corspec_base, TLB_GET_SIZE=size

  delx = size[0]-state.buffer[0]-(state.specplotsize[0]+state.repplotsize[0])
  ysize = size[1]-state.buffer[1]

  state.specplotsize[0] = state.specplotsize[0]+delx/2
  state.specplotsize[1] = ysize

  state.repplotsize[0] = state.repplotsize[0]+delx/2
  state.repplotsize[1] = ysize
  
  widget_control, state.corspec_base,UPDATE=0
  widget_control, state.plotwin1, DRAW_XSIZE=state.specplotsize[0], $
                  DRAW_YSIZE=state.specplotsize[1]

  widget_control, state.plotwin2, DRAW_XSIZE=state.repplotsize[0], $
                  DRAW_YSIZE=state.repplotsize[1]

  widget_control, state.corspec_base,UPDATE=1
   
  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.specplotsize[0],YSIZE=state.specplotsize[1]
  state.pixmap_wid = !d.window

  corspec_plotupdate,state,/REPORT
  
;  if state.freeze then begin
;
;     erase, COLOR=20
;     wset, state.plotwin_wid
;     device, COPY=[0,0,state.plotwin_size[0],state.plotwin_size[1],0,0,$
;                   state.pixmap_wid]
;     
;  endif else xmergeorders_plotspec

  widget_control, state.corspec_base, SET_UVALUE=state, /NO_COPY
  
end  
;
;===============================================================================
;
;------------------------------- Main Program ----------------------------------
;
;===============================================================================
;
pro xmc_corspec,xanchor,yanchor,x,y,offset,JUSTFIT=justfit,MAX=max,$
                TITLE=title,ANCHORLABEL=anchorlabel,TESTLABEL=testlabel, $
                CANCEL=cancel

  cancel = 0
   
  cleanplot,/SILENT

  mc_mkct

  mc_getfonts,buttonfont,textfont

  anchorlabel = keyword_set(ANCHORLABEL) ? anchorlabel:'Disk Spectrum'
  testlabel = keyword_set(TESTLABEL) ? testlabel:'Observed Spectrum'  

; remove NaNs from the data

  idx = mc_nantrim(yanchor,2)
  xxanchor = xanchor[idx]
  yyanchor = yanchor[idx]

  idx = mc_nantrim(y,2)
  xx = x[idx]
  yy = y[idx]

;  Now find the regions of common x value

  match,xxanchor,xx,anchoridx,idx

  xxanchor = xxanchor[anchoridx]
  yyanchor = yyanchor[anchoridx]

  xx = xx[idx]
  yy = yy[idx]
    
  ptroffset = ptr_new(2.0)
  ptrcancel = ptr_new(0)

  state = {absxrange:[0.,0.],$
           absyrange:[0.,0.],$
           anchorlabel:anchorlabel,$
           buffer:[0L,0L],$
           cancel:ptrcancel,$
           charsize:1.5,$
           corr:ptr_new(2),$
           corrange:mc_bufrange([min(xxanchor,MAX=xmax,/NAN),xmax],-0.05),$
           corspec_base:0L,$
           cursormode:'None',$
           fit:ptr_new(2),$
           lag:ptr_new(2),$
           max:keyword_set(MAX),$
           offset:ptroffset,$
           pixmap_wid:0L,$
           plotwin1:0L,$
           plotwin1_wid:0L,$
           plotwin2:0L,$
           plotwin2_wid:0L,$
           pscale:!p,$
           reg:[[!values.d_nan,!values.d_nan],$
                [!values.d_nan,!values.d_nan]],$
           repplotsize:[300,400],$
           specplotsize:[600,400],$
           testlabel:testlabel,$
           thick:1,$
           title:(n_elements(TITLE) ne 0) ? title:'',$
           x:xx,$
           xanchor:xxanchor,$
           xmax_fld:[0L,0L],$
           xmin_fld:[0L,0L],$
           xrange:[0.,0.],$
           xscale:!x,$
           y:yy/mean(yy),$
           yanchor:yyanchor/mean(yyanchor),$
           ymax_fld:[0L,0L],$
           ymin_fld:[0L,0L],$
           yrange:[0.,0.],$
           yscale:!y}

  corspec_corspec,state

  if keyword_set(JUSTFIT) then begin

     cancel = *ptrcancel
     offset = (cancel eq 1) ? -1:*ptroffset
     
     ptr_free, ptrcancel
     ptr_free, ptroffset
     state = 0B
     return
     
  endif
  
;  Build the widget.

  state.corspec_base = widget_base(TITLE='XCorrelate Spectra', $
                                   /COLUMN,$
                                   /TLB_SIZE_EVENTS)

     button = widget_button(state.corspec_base,$
                            FONT=buttonfont,$
                            EVENT_PRO='corspec_event',$
                            VALUE='Cancel',$
                            UVALUE='Cancel')

     row_base = widget_base(state.corspec_base,$
                            /ROW,$
                            /BASE_ALIGN_CENTER)

           state.plotwin1 = widget_draw(row_base,$
                                        XSIZE=state.specplotsize[0],$
                                        YSIZE=state.specplotsize[1],$
                                        /TRACKING_EVENTS,$
                                        /BUTTON_EVENTS,$
                                        /MOTION_EVENTS,$
                                        /KEYBOARD_EVENTS,$
                                        EVENT_PRO='corspec_plotwinevent',$
                                        UVALUE='Plot Window 1')
           
           state.plotwin2 = widget_draw(row_base,$
                                        XSIZE=state.repplotsize[0],$
                                        YSIZE=state.repplotsize[1],$
                                        UVALUE='Plot Window 2')

     row = widget_base(state.corspec_base,$
                       /ROW,$
                       /BASE_ALIGN_LEFT,$
                       FRAME=2)
     
        xmin = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='X Min:',$
                             UVALUE='X Min',$
                             XSIZE=12,$
                             EVENT_PRO='corspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmin_fld = [xmin,textid]
                
        xmax = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='X Max:',$
                             UVALUE='X Max',$
                             XSIZE=12,$
                             EVENT_PRO='corspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.xmax_fld = [xmax,textid]
        
        ymin = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='Y Min:',$
                             UVALUE='Y Min',$
                             XSIZE=12,$
                             EVENT_PRO='corspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymin_fld = [ymin,textid]
        
        ymax = coyote_field2(row,$
                             LABELFONT=buttonfont,$
                             FIELDFONT=textfont,$
                             TITLE='Y Max:',$
                             UVALUE='Y Max',$
                             XSIZE=12,$
                             EVENT_PRO='corspec_minmaxevent',$
                             /CR_ONLY,$
                             TEXTID=textid)
        state.ymax_fld = [ymax,textid]

     button = widget_button(state.corspec_base,$
                            FONT=buttonfont,$
                            EVENT_PRO='corspec_event',$
                            VALUE='Accept',$
                            UVALUE='Accept')
        
           
; Get things running.  Center the widget using the Fanning routine.
           
  cgcentertlb,state.corspec_base
  widget_control, state.corspec_base, /REALIZE
           
;  Get plotwin ids
           
  widget_control, state.plotwin1, GET_VALUE=val
  state.plotwin1_wid = val
  
  window, /FREE, /PIXMAP,XSIZE=state.specplotsize[0],YSIZE=state.specplotsize[1]
  state.pixmap_wid = !d.window

  widget_control, state.plotwin2, GET_VALUE=val
  state.plotwin2_wid = val
  
;  Get sizes for things.
           
  widget_geom = widget_info(state.corspec_base, /GEOMETRY)
 
  state.buffer[0]=widget_geom.xsize-(state.specplotsize[0]+state.repplotsize[0])
  state.buffer[1]=widget_geom.ysize-state.specplotsize[1]

;  Get plot ranges

  state.xrange = [min(xxanchor,MAX=xmax,/NAN),xmax]
  state.absxrange = state.xrange

;  Smooth the data to avoid bad pixels
  
  xx = findgen(n_elements(state.xanchor))
  syanchor = mc_robustsg(xx,state.yanchor,5,3,0.1,CANCEL=cancel)
  if cancel then return

  xx = findgen(n_elements(state.x))
  sy = mc_robustsg(xx,state.y,5,3,0.1,CANCEL=cancel)
  if cancel then return
  
  min = min([syanchor[*,1],sy[*,1]],/NAN,MAX=max)
  state.absyrange = mc_bufrange([min,max],0.1,CANCEL=cancel)
  if cancel then return
  state.yrange = state.absyrange

  corspec_setminmax,state

;  Plot the spectra

  corspec_plotupdate,state,/REPORT
    
; Put state variable into the user value of the top level base.

  base = state.corspec_base
  widget_control, state.corspec_base, SET_UVALUE=state, /NO_COPY
  
; Start the Event Loop. This will be a non-blocking program.

  XManager, 'corspec', $
            base,$
            EVENT_HANDLER='corspec_resizeevent',$
            CLEANUP='corspec_cleanup'

  cancel = *ptrcancel
  offset = (cancel eq 1) ? -1:*ptroffset

  ptr_free, ptrcancel
  ptr_free, ptroffset
  state = 0B

  
end
  
