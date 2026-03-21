;+
; NAME:
;     xmc_plotprofiles
;    
; PURPOSE:
;     Plots spatial profiles. 
;
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xmc_plotprofiles,profiles,orders,doorders,slith_arc,AUTONAP=autonap,$
;                      GUESSNAP=guessnap,FIXNAP=fixnap,APPOS=appos,$
;                      GUESSPOS=guesspos,APRADII=apradii,MASK=mask,$
;                      PSFAP=psfap,POSITION=position,GETINFO=getinfo,$
;                      GROUP_LEADER=group_leader,LOCK=lock,CANCEL=cancel
; INPUTS:
;     profiles  - A structure with [norder] elements where 
;                 struct.(i) = [[arcseconds],[data]] is the x and y
;                 values for the ith order
;     orders    - An array [norders] of order numbers
;     doorders  - An array [norders] if zeros and ones denoting
;                 whether an order should be plotted or not.
;     slith_arc - Slit length in arcseconds
;    
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     AUTONAP      - The number of apertures to automatically search.
;     GUESSNAP     - The number of guess aperture positions the user will
;                    designate.
;     FIXNAP       - The number of apertures positions the user will designate.
;     APPOS        - An array [naps,norders] of aperture positions in 
;                    arcseconds
;     GUESSPOS     - An array [naps,norders] of guess positions that
;                    will be used to identify aperture positions.
;     APRADII      - An array [naps] of aperture radii
;     MASK         - An array [norders] of structures {x_values,mask}
;                    giving the mask.  (See mkmask_ps.pro)
;     PSFAP        - The PSF radius.
;     POSITION     - An 2-element array giving the position the window
;                    is to be displayed.  [0,0] is the ULH corner.
;     GETINFO      - A structure will be returned with pertinent
;                    information:
;                     {doorders:doorders,appos:appos,apsign:apsign,
;                     guesspos:guesspos}
;     LOCK         - If set, ther user cannot select/unselect orders.
;     GROUP_LEADER - The widget ID of an existing widget that serves
;                    as "group leader" for the newly-created widget. 
;     CANCEL       - Set on return if there is a problem
;     
; OUTPUTS:
;     None
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     xmc_plotprofiles_state
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     The resizable widget just plots the superprofiles.  The user can
;     interact with the widget depending on which keywords are passed.
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     ? - Written by M. Cushing, Institute for Astronomy, UH
;     2003-06-15 - Added Output Profiles button
;     2005-08-13 - Modified to generate to xspextool.pro an event when
;                  the apertures are selected.
;     2016-01    - Massive rewrite to accomidate new method of
;                  choosing apertures and orders.
;-
;
;===============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;===============================================================================
;
pro plotprofiles_startup,profiles,orders,doorders,slith_arc,POSITION=position, $
                         GROUP_LEADER=group_leader

  common plotprofiles_state, state

  screensize = Get_Screen_Size()

  
  cleanplot,/SILENT
  
  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

  state = {appos:ptr_new(fltarr(2)),$
           apradii:ptr_new(2),$
           buffer:[0L,0L],$
           but_edgemask:ptr_new(0),$
           but_order:0L,$
           but_ord:ptr_new(0),$
           buttonfont:buttonfont,$
           charsize:1.5,$
           cursormode:'None',$
           doorders:ptr_new(intarr(1)),$
           edgemask:1,$
           guesspos:ptr_new(fltarr(2)),$
           lock:0L,$
           mask:ptr_new(fltarr(1)),$
           message:0L,$
           minplotwinsize:300,$
           mode:'Plot',$
           naps:0,$
           norders:n_elements(orders),$
           oidx:0,$
           orders:ptr_new(intarr(2)),$
           origprofiles:ptr_new(fltarr(2,2)),$
           pixpp:250.0,$
           plotwin_wid:0L,$
           pixmap_wid:0L,$
           plotappos:0,$
           plotguesspos:0,$
           plotmask:0,$
           plotpsf:0,$
           plotwin:0L,$
           plotwinsize:[screensize[0]*0.35,0],$
           profiles:ptr_new(fltarr(2,2)),$
           pscale:!p,$
           psfap:0.0,$
           reg:[[!values.f_nan,!values.f_nan],$
                [!values.f_nan,!values.f_nan]],$
           scrollsize:[screensize[0]*0.35,600],$
           select_but:0L,$
           textfont:textfont,$

           xmc_plotprofiles_base:0L,$
           xrange:[0.,slith_arc],$
           xscale:!x,$
           yscale:!y}

;  Load user data that can change in size

  *state.orders       = orders
  *state.origprofiles = profiles
  *state.profiles     = profiles
  *state.doorders     = doorders
  
;  Set up size of window and plot area

  state.plotwinsize[1] = state.minplotwinsize > state.pixpp*state.norders
  state.scrollsize[1] = (state.plotwinsize[1] gt screensize[1]) ? $
                        screensize[1]*0.75:state.plotwinsize[1]
  
;  Make the widget
  
  state.xmc_plotprofiles_base = widget_base(TITLE='Spatial Profiles',$
                                              MBAR=mbar,$
                                              /COLUMN,$
                                              GROUP_LEADER=group_leader,$
                                              /TLB_SIZE_EVENTS)
  
     button = widget_button(mbar, $
                            VALUE='File', $
                            /MENU,$
                            EVENT_PRO='plotprofiles_event',$
                            FONT=buttonfont)
     
        quit = widget_button(button, $
                             VALUE='Quit',$
                             UVALUE='Quit',$
                             FONT=buttonfont)

     state.but_order = widget_button(mbar, $
                                       VALUE='View Order', $
                                       UVALUE='Order Menu',$
                                       EVENT_PRO='plotprofiles_event',$
                                       /MENU,$
                                       FONT=buttonfont)

        (*state.but_ord) = lonarr(n_elements(orders))
        for i = n_elements(orders)-1,0,-1 do begin

           val =  strtrim('Order '+string(orders[i],FORMAT='(I3)'),2)
           (*state.but_ord)[i] = widget_button(state.but_order,$
                                                 VALUE=val,$
                                                 UVALUE='Order Menu',$
                                                 FONT=state.buttonfont)
        
     endfor

        
     state.select_but = widget_button(mbar, $
                                        VALUE='Select Orders', $
                                        /MENU,$
                                        EVENT_PRO='plotprofiles_event',$
                                        FONT=buttonfont)
     
        all = widget_button(state.select_but, $
                            VALUE='All Orders',$
                            UVALUE='All Orders',$
                            FONT=buttonfont)
        
        none = widget_button(state.select_but, $
                             VALUE='No Orders',$
                             UVALUE='No Orders',$
                             FONT=buttonfont)

     mask_but = widget_button(mbar, $
                              VALUE='Edge Mask', $
                              /MENU,$
                              /NO_RELEASE,$
                              EVENT_PRO='plotprofiles_event',$
                              FONT=buttonfont)

     values = ['None','1 Pixel','2 Pixels','3 Pixels','4 Pixels','5 Pixels']
     (*state.but_edgemask) = lonarr(n_elements(values))
     for i = 0,n_elements(values)-1 do begin

        
        (*state.but_edgemask)[i] = widget_button(mask_but, $
                                                   /CHECKED_MENU,$
                                                   VALUE=values[i],$
                                                   UVALUE='Edge Mask',$
                                                   FONT=buttonfont)
     endfor
     widget_control, (*state.but_edgemask)[1],/SET_BUTTON
     
     state.message = widget_label(state.xmc_plotprofiles_base,$
                                    VALUE=' Cursor X :',$
                                    /ALIGN_LEFT,$
                                    FONT=buttonfont,$
                                    /DYNAMIC_RESIZE)
            
;  Get window size
   
     if state.plotwinsize[1] le state.scrollsize[1] then begin
        
        state.plotwin = widget_draw(state.xmc_plotprofiles_base,$
                                      XSIZE=state.plotwinsize[0],$
                                      YSIZE=state.plotwinsize[1],$
                                      /TRACKING_EVENTS,$
                                      /MOTION_EVENTS,$
                                      /BUTTON_EVENTS,$
                                      /KEYBOARD_EVENTS,$
                                      UVALUE='Plot Window',$
                                      EVENT_PRO='plotprofiles_plotwinevent')
        
     endif else begin
        
        state.plotwin = widget_draw(state.xmc_plotprofiles_base,$
                                      XSIZE=state.plotwinsize[0],$
                                      YSIZE=state.plotwinsize[1],$
                                      X_SCROLL_SIZE=state.scrollsize[0],$
                                      Y_SCROLL_SIZE=state.scrollsize[1],$
                                      /TRACKING_EVENTS,$
                                      /MOTION_EVENTS,$
                                      /BUTTON_EVENTS,$
                                      /KEYBOARD_EVENTS,$
                                      UVALUE='Plot Window',$
                                      /SCROLL,$
                                      EVENT_PRO='plotprofiles_plotwinevent')
     
     endelse          
     
     if n_elements(POSITION) eq 0 then position = [0.5,0.5]
     cgcentertlb,state.xmc_plotprofiles_base,position[0],position[1]
     
     widget_control, state.xmc_plotprofiles_base, /REALIZE
     
;  Get plotwin ids

   widget_control, state.plotwin, GET_VALUE=x
   state.plotwin_wid = x

   window, /FREE, /PIXMAP,XSIZE=state.plotwinsize[0],$
     YSIZE=state.plotwinsize[1]
   state.pixmap_wid = !d.window
   
; Start the Event Loop. This will be a non-blocking program.
   
   XManager, 'xmc_plotprofiles', $
     state.xmc_plotprofiles_base, $
     /NO_BLOCK,$
     EVENT_HANDLER='plotprofiles_resizeevent',$
     CLEANUP='plotprofiles_cleanup'
   
   geom = widget_info(state.xmc_plotprofiles_base, /geometry)
   
   state.buffer[0] = geom.xsize-state.scrollsize[0]
   state.buffer[1] = geom.ysize-state.scrollsize[1]
     
end
;
;==============================================================================
;
pro plotprofiles_edgemask

  common plotprofiles_state

  if state.edgemask eq 0 then begin

     *state.profiles = *state.origprofiles
     return
     
  endif
     
  for i = 0, state.norders-1 do begin
     
     prof = (*state.origprofiles).(i)
     n = n_elements(prof[*,0])
     prof[0:state.edgemask,1] = !values.f_nan
     prof[(n-state.edgemask-1):(n-1),1] = !values.f_nan

     (*state.profiles).(i) = prof
     
  endfor
   
end
;
;===============================================================================
;
pro plotprofiles_modwinsize,DESTROY=destroy

  common plotprofiles_state
  
;  Modify plot window according to the number of orders
  
  widget_control, state.xmc_plotprofiles_base, UPDATE=0
  if keyword_set(DESTROY) then widget_control, state.plotwin, /DESTROY
  
  if state.plotwinsize[1] le state.scrollsize[1] then begin
     
     state.plotwin = widget_draw(state.xmc_plotprofiles_base,$
                                   XSIZE=state.plotwinsize[0],$
                                   YSIZE=state.plotwinsize[1],$
                                   UVALUE='Plot Window',$
                                   /TRACKING_EVENTS,$
                                   /MOTION_EVENTS,$
                                   /BUTTON_EVENTS,$
                                   /KEYBOARD_EVENTS,$
                                   EVENT_PRO='plotprofiles_plotwinevent')
     
  endif else begin
     
     state.plotwin = widget_draw(state.xmc_plotprofiles_base,$
                                   XSIZE=state.plotwinsize[0],$
                                   YSIZE=state.plotwinsize[1],$
                                   X_SCROLL_SIZE=state.scrollsize[0],$
                                   Y_SCROLL_SIZE=state.scrollsize[1],$
                                   UVALUE='Plot Window',$
                                   /SCROLL,$
                                   /TRACKING_EVENTS,$
                                   /MOTION_EVENTS,$
                                   /BUTTON_EVENTS,$
                                   /KEYBOARD_EVENTS,$
                                   EVENT_PRO='plotprofiles_plotwinevent')
     
  endelse
  
  widget_control, state.xmc_plotprofiles_base, UPDATE=1
  
  geom = widget_info(state.xmc_plotprofiles_base,/GEOMETRY)
  
  state.buffer[0]=geom.xsize-state.scrollsize[0]
  state.buffer[1]=geom.ysize-state.scrollsize[1] 
  
  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plotwinsize[0],YSIZE=state.plotwinsize[1]

  state.pixmap_wid = !d.window
  
end
;
;===============================================================================
;
pro plotprofiles_cleanup,event

  common plotprofiles_state
  
  ptr_free, state.but_ord
  ptr_free, state.doorders
  ptr_free, state.mask
  ptr_free, state.but_edgemask
  ptr_free, state.orders
  ptr_free, state.appos
  ptr_free, state.profiles
  ptr_free, state.guesspos
  
  state = 0B
  
end
;
;===============================================================================
;
pro plotprofiles_plotupdate

  common plotprofiles_state
  
  wset, state.pixmap_wid
  erase
  polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM
  plotprofiles_plotprofiles
  
  state.pscale = !p
  state.xscale = !x
  state.yscale = !y
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plotwinsize[0],state.plotwinsize[1],0,0,$
                state.pixmap_wid]
  
  
end
;
;===============================================================================
;
pro plotprofiles_plotprofiles 

  common plotprofiles_state
  
  !p.multi[2] = state.norders
  !p.multi[0] = state.norders
  
  charsize = state.charsize
  if state.norders gt 2 then charsize = charsize*2.0
  
  for i = 0, state.norders-1 do begin
     
     j = state.norders-1-i
     prof = (*state.profiles).(j)
     
;  Get plot range.
     
     ymin = min(prof[*,1],max=ymax,/NAN)

     color = ((*state.doorders)[j] eq 0) ? 100:1
     plot,prof[*,0],prof[*,1],CHARSIZE=charsize,XRANGE=state.xrange,$
          XTITLE='!5Slit Position (arcsec)',/XSTY,PSYM=10,$
          TITLE='!5Order '+strtrim(string((*state.orders)[j], $
                                          FORMAT='(I3)'),2),$
          YTITLE='!5Relative Flux',/YSTY,YRANGE=mc_bufrange([ymin,ymax],0.05), $
          /NODATA,COLOR=color
     
     oplot, prof[*,0],prof[*,1],PSYM=10,COLOR=color
     
     if state.plotappos then begin
        
        if (*state.doorders)[j] then begin
           
           for k = 0, state.naps-1 do begin

              plots,[(*state.appos)[k,j],(*state.appos)[k,j]],!y.crange,$
                    COLOR=7

           endfor
              
        endif
        
     endif
     
     if state.plotmask then begin
        
        mask = (*state.mask).(j)
        if (*state.doorders)[j] then begin
           
;  Plot apertures
           
           if finite((*state.apradii)[0]) eq 1 then begin
              
              z = where(mask[*,1] le 0.0)
              tmp = prof[*,1]
              tmp[z] = !values.f_nan
              oplot,prof[*,0],tmp,PSYM=10,COLOR=3
              
           endif
           
;  Plot BG regions
           
           z = where(mask[*,1] ge 0, count)
           if count eq 0 then goto, cont
           tmp = prof[*,1]
           tmp[z] = !values.f_nan
           oplot,prof[*,0],tmp,PSYM=10,COLOR=2
           
           cont:
           
;  Plot apertures 
           
           if finite((*state.apradii)[0]) eq 1 then begin
              
              for k = 0, state.naps-1 do begin
                 
                 plots,[(*state.appos)[k,j],(*state.appos)[k,j]]- $
                       (*state.apradii)[k],!y.crange,LINESTYLE=1,COLOR=3
                 plots,[(*state.appos)[k,j],(*state.appos)[k,j]]+ $
                       (*state.apradii)[k],!y.crange,LINESTYLE=1,COLOR=3
                 
              endfor
              
           endif
           
        endif
        
     endif
     
     if state.plotpsf then begin
        
        if (*state.doorders)[j] then begin
           
           for k = 0, state.naps-1 do begin
              
              plots,[(*state.appos)[k,j]-state.psfap,$
                     (*state.appos)[k,j]-state.psfap],!y.crange,COLOR=4,$
                    LINESTYLE=1
              plots,[(*state.appos)[k,j]+state.psfap,$
                     (*state.appos)[k,j]+state.psfap],!y.crange,COLOR=4,$
                    LINESTYLE=1
              
           endfor
           
        endif
        
     endif

     if state.plotguesspos then begin
     
        if (*state.doorders)[j] then begin
           
           for k = 0,state.naps-1 do begin
              
              plots,replicate((*state.guesspos)[k,j],2),!y.crange, $
                    LINESTYLE=2,COLOR=7
              
           endfor
           
        endif
        
     endif
     
  endfor
  
  !p.multi=0

end
;
;===============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;===============================================================================
;
pro plotprofiles_event,event

  common plotprofiles_state
  
  widget_control, event.id,  GET_UVALUE = uvalue
  widget_control, /HOURGLASS
  
  case uvalue of 

     'Edge Mask': begin
        
        z = where(*state.but_edgemask eq event.id)
        state.edgemask = z[0]
        
        for i = 0,5 do widget_control, (*state.but_edgemask)[i],SET_BUTTON=0
        widget_control, (*state.but_edgemask)[state.edgemask],SET_BUTTON=1
        plotprofiles_edgemask
        plotprofiles_plotupdate        

     end
          
     'Order Menu': begin

        if state.plotwinsize[1] ne state.scrollsize[1] then begin
        
           z = where(*state.but_ord eq event.id)
           
           del = max(*state.orders,MIN=min)-min+1
           offset = state.plotwinsize[1]/float((del+1))
           frac = ((*state.orders)[z]-min)/float(del)
           
           widget_control, state.plotwin, $
                           SET_DRAW_VIEW=[0,state.plotwinsize[1]*frac-offset]

        endif
        
     end
     
     'All Orders': begin

        (*state.doorders)[*] = 1              
        plotprofiles_plotupdate

     end

     'No Orders': begin
        
        (*state.doorders)[*] = 0              
        plotprofiles_plotupdate
        
     end
     
     'Quit': begin
        
        widget_control, event.top, /DESTROY
        !p.multi=0
        
     end

  endcase
  
end
;
;===============================================================================
;
pro plotprofiles_plotwinevent,event

  common plotprofiles_state
  
  widget_control, event.id,  GET_UVALUE = uvalue
  
;  Check to see if it is a TRACKING event.
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin
     
     wset, state.plotwin_wid
     device, COPY=[0,0,state.plotwinsize[0],state.plotwinsize[1],0,0,$
                   state.pixmap_wid]
     widget_control, state.plotwin,INPUT_FOCUS=event.enter
     goto, cont
     
  endif

  !p = state.pscale
  !x = state.xscale
  !y = state.yscale
  x  = event.x/float(state.scrollsize[0])
  y  = event.y/float(state.scrollsize[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA)

;  Check for arrow keys

  if event.type eq 6 and event.release eq 0 then begin

     widget_control, state.plotwin,GET_DRAW_VIEW=current
     offset = state.plotwinsize[1]/state.norders
     max = state.plotwinsize[1]-state.scrollsize[1]

     case event.key of

        7: begin ; up
           
           val = (current[1]+offset) < max
           widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
                      
        end

        8: begin ;  down

           val = (current[1]-offset) > 0
           widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
           
        end

        else:

     endcase

  endif

;  Are we locked out?

  if state.lock then goto, out

;  Find which order the cursor is over
  
  state.oidx = floor(event.y/float(state.plotwinsize[1])*state.norders)
  
;  Is it a keyboard event?

  if event.ch ne 0 and event.release then begin

     case event.ch of 

        113: begin  ; q

           widget_control, event.top, /DESTROY
           !p.multi=0
           return
           
        end
        
        97: begin  ; a
           
           (*state.doorders)[*] = 1
           plotprofiles_plotupdate
           
        end
        
        110: begin  ; n
           
           (*state.doorders)[*] = 0              
           plotprofiles_plotupdate
           
        end

        115: begin  ; s

           (*state.doorders)[state.oidx] = $
              ~(*state.doorders)[state.oidx] 
           plotprofiles_plotupdate
           
        end

        109: begin              ; m

           if state.mode ne 'Auto' then begin
              
              state.cursormode = 'Modify'
              state.reg= !values.f_nan
              
              if state.mode eq 'Guess' then begin
                 
                 state.plotguesspos = 1
                 state.plotappos = 0
                 
              endif
              
           endif
           
        end                     
        
        else:
        
     endcase

     plotprofiles_plotupdate                
     goto, cont

  endif
 
;  If not, it is some kind of plot window event
  
  if event.type eq 2 then goto, out
  
  case state.cursormode of 

     'Guess': begin
        
        if event.type ne 1 then goto, out        
        z = where(finite((*state.guesspos)[*,0]) eq 0,cnt)
        if cnt ge 1 then (*state.guesspos)[z[0],*] = xy[0]
        if cnt eq 1 then begin
           
           mc_findpeaks,*state.profiles,state.naps, $
                        make_array(state.norders,/INTEGER,VALUE=1), $
                        positions,GUESS=*state.guesspos,CANCEL=cancel
           if cancel then return
           
           state.plotappos     = 1
           state.plotguesspos = 1
           *state.appos      = positions                            
           state.cursormode  = 'None'
           
        endif
        
        plotprofiles_plotupdate
        
     end

     'Fix': begin

        if event.type ne 1 then goto, out                
        z = where(finite((*state.appos)[*,0]) eq 0,cnt)
        if cnt ge 1 then (*state.appos)[z[0],*] = xy[0]
        if cnt eq 1 then begin

           mc_findpeaks,*state.profiles,state.naps, $
                        make_array(state.norders,/INTEGER,VALUE=1), $
                        positions,FIXED=*state.appos,CANCEL=cancel
           if cancel then return

           state.plotappos     = 1
           state.plotguesspos = 0
           state.cursormode  = 'None'
           
        endif
        
        plotprofiles_plotupdate
        
     end
        
     'Modify': begin
        
        case state.mode of
           
           'Guess': begin
              
              case event.type of
                 
                 0: begin
                    
                    vals = (*state.guesspos)[*,state.oidx]
                    min = min(abs(xy[0]-vals),idx)
                    (*state.guesspos)[idx,state.oidx] = !values.f_nan
                    
                 end
                 
                 1: begin

                    z = where(finite(*state.guesspos) eq 0,cnt)
                    (*state.guesspos)[z] = xy[0]
                    state.cursormode = 'None'

                    mc_findpeaks,*state.profiles,state.naps, $
                                 make_array(state.norders,/INTEGER,VALUE=1), $
                                 positions,GUESS=*state.guesspos, $
                                 CANCEL=cancel
                    if cancel then return

                    state.plotappos = 1
                    state.plotguesspos = 1
                    *state.appos      = positions                            
                    state.cursormode  = 'None'

                 end
                 
                 else:
                 
              endcase

              plotprofiles_plotupdate
              
           end
           
           'Fix': begin

              case event.type of
                 
                 0: begin
                    
                    vals = (*state.appos)[*,state.oidx]
                    min = min(abs(xy[0]-vals),idx)
                    (*state.appos)[idx,state.oidx] = !values.f_nan

                 end
                 
                 1: begin

                    z = where(finite(*state.appos) eq 0,cnt)
                    (*state.appos)[z] = xy[0]
                    state.cursormode = 'None'
                    state.plotappos     = 1

                 end

                 else:
                 
              endcase

              plotprofiles_plotupdate
                           
           end

           'Plot': begin

              case event.type of
                 
                 0: begin
                    
                    vals = (*state.appos)[*,state.oidx]
                    min = min(abs(xy[0]-vals),idx)
                    (*state.appos)[idx,state.oidx] = !values.f_nan

                 end
                 
                 1: begin

                    z = where(finite(*state.appos) eq 0,cnt)
                    (*state.appos)[z] = xy[0]
                    state.cursormode = 'None'
                    state.plotappos     = 1

                 end

                 else:
                 
              endcase

              plotprofiles_plotupdate
                           
           end

        endcase

     end

     else:
     
  endcase

  out:
  
;  Update cursor position.
  
  label = ' Cursor X : '+strtrim(xy[0],2)+' arcsec'
  widget_control,state.message,SET_VALUE=label
  
;  Copy the pixmaps and draw the lines.
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plotwinsize[0],state.plotwinsize[1],0,0,$
                state.pixmap_wid]

  case state.cursormode of

     'Modify': begin

        low = state.oidx*(state.plotwinsize[1]/state.norders)
;        plots,[event.x,event.x],[0,low],COLOR=2,/DEVICE
        plots,[event.x,event.x], $
              [low,low+state.plotwinsize[1]/state.norders], $
              COLOR=2,/DEVICE
;        plots,[event.x,event.x], $
;              [low+state.plotwinsize[1]/state.norders,$
;               state.plotwinsize[1]],COLOR=2,/DEVICE
       
     end

     'Guess': plots, [event.x,event.x], $
                     [0,state.plotwinsize[1]],COLOR=7,/DEVICE

     'Fix': plots, [event.x,event.x], $
                     [0,state.plotwinsize[1]],COLOR=7,/DEVICE

     else: plots, [event.x,event.x],[0,state.plotwinsize[1]],COLOR=2,/DEVICE 

  endcase
  
cont:

end
;
;===============================================================================
;
pro plotprofiles_resizeevent, event

  common plotprofiles_state
  
  widget_control, state.xmc_plotprofiles_base, TLB_GET_SIZE = size
  
  state.plotwinsize[0] = size[0]-state.buffer[0]
  state.scrollsize[0]  = state.plotwinsize[0]
  
  state.scrollsize[1]  = size[1]-state.buffer[1]
  state.plotwinsize[1] = state.scrollsize[1] > $
                           state.pixpp*state.norders
  
  plotprofiles_modwinsize,/DESTROY
  plotprofiles_plotupdate
  
end
;
;===============================================================================
;
; -------------------------------Main Program---------------------------------
;
;===============================================================================
;
pro xmc_plotprofiles,profiles,orders,doorders,slith_arc,AUTONAP=autonap, $
                     GUESSNAP=guessnap,FIXNAP=fixnap,APPOS=appos, $
                     GUESSPOS=guesspos,APRADII=apradii,MASK=mask,PSFAP=psfap, $
                     POSITION=position,GETINFO=getinfo, $
                     GROUP_LEADER=group_leader,LOCK=lock,CANCEL=cancel
  

  cancel = 0

  common plotprofiles_state
  
;  Collecting information?  If so, collect and return
  
  if arg_present(GETINFO) ne 0 then begin

;  Summarize the apsigns

     apsign = mc_getapsign(*state.profiles,*state.appos,*state.doorders, $
                           CANCEL=cancel)
     if cancel then return


     if size(apsign,/N_DIMENSION) gt 1 then apsign = total(apsign,2)/ $
        total(abs(apsign),2)

     for i = 0, state.naps-1 do apsign[i] = (apsign[i] ge 0) ? 1:-1

;  Construct the output structure
     
     getinfo = {doorders:*state.doorders,appos:*state.appos, $
                apsign:fix(apsign),guesspos:*state.guesspos}
     return
     
  endif

;  Has the widget been generated?  If not, do so.
  
  if not xregistered('xmc_plotprofiles') then begin
     
     plotprofiles_startup,profiles,orders,doorders,slith_arc, $
                          POSITION=position,GROUP_LEADER=group_leader
          
  endif else begin

;  Load user data

     state.norders       = n_elements(orders)
     *state.orders       = orders
     *state.origprofiles = profiles
     *state.profiles     = profiles
     *state.doorders     = doorders
     state.xrange        = [0,slith_arc]              
     
     state.plotappos  = 0
     state.plotmask = 0
     state.plotpsf  = 0
     state.plotguesspos = 0
     
     state.cursormode = 'None'
     state.mode = 'Plot'
       
  endelse

;  Update order button if need be

  if total(*state.orders) ne total(orders) then begin

;  Destroy old buttons
     
     for i = 0,n_elements(*state.but_ord)-1 do begin
        
        widget_control, (*state.but_ord)[i], /DESTROY
        
     endfor

;  Create new buttons
     
     (*state.but_ord) = lonarr(n_elements(orders))
     
     for i = 0,n_elements(orders)-1 do begin
        
        (*state.but_ord)[i] = widget_button(state.but_order,$
                                              VALUE='Order '+string(orders[i], $
                                                    FORMAT='(I3.3)'),$
                                              UVALUE='Order Menu',$
                                              FONT=state.buttonfont)
        
     endfor
     
     plotprofiles_modwinsize,/DESTROY
  
  endif

;  Now edge-mask the profiles

  plotprofiles_edgemask

;  Are we locked?
  
  state.lock = keyword_set(LOCK)
  widget_control, state.select_but, SENSITIVE=~state.lock

;  Find apertures automatically 
  
  if n_elements(AUTONAP) then begin

     state.naps = autonap
     mc_findpeaks,*state.profiles,state.naps, $
                  make_array(state.norders,/INTEGER,VALUE=1), $
                  positions,apsign,/AUTO,CANCEL=cancel
     if cancel then return

     state.plotappos    = 1
     state.plotguesspos = 0
     *state.appos       = positions
     state.cursormode   = 'None'
     state.mode         = 'Auto'
     
  endif

;  Ask user to identify guess positions
  
  if n_elements(GUESSNAP) then begin

     state.mode = 'Guess'
     state.cursormode = 'Guess'
     state.plotguesspos = 1
     state.naps = guessnap
     *state.guesspos = fltarr(guessnap,state.norders)+!values.f_nan
     
  endif

;  Fix apertures
  
  if n_elements(FIXNAP) then begin
     
     state.mode = 'Fix'
     state.cursormode = 'Fix'
     state.plotappos = 1
     state.plotguesspos = 0
     state.naps = fixnap
     *state.appos = fltarr(fixnap,state.norders)+!values.f_nan
     
  endif

;  Do we plot aperture positions?
  
  if n_elements(APPOS) ne 0 then begin

     state.naps      = (size(APPOS))[1]
     state.plotappos = 1
     *state.appos    = appos
     
  endif

;  Find apertures based on guess positions
  
  if n_elements(GUESSPOS) ne 0 then begin
     
     state.plotguesspos   = 1
     *state.guesspos      = guesspos

     mc_findpeaks,*state.profiles,state.naps, $
                  make_array(state.norders,/INTEGER,VALUE=1), $
                  positions,GUESS=*state.guesspos,CANCEL=cancel
     if cancel then return
     
     state.plotappos     = 1
     state.plotguesspos = 1
     *state.appos      = positions                            
     state.cursormode  = 'None'
          
  endif 

; Load the aperture radii
  
  if n_elements(APRADII) ne 0 then *state.apradii = apradii else $
     *state.apradii = !values.f_nan

;  Do we plot the slit mask?
  
  if n_elements(MASK) ne 0 then begin
     
     state.plotmask = 1
     *state.mask = mask
     
  endif

;  Do we plot the PSF radius?
  
  if n_elements(PSFAP) ne 0 then begin
     
     state.psfap = psfap
     state.plotpsf = 1
     
  endif

;  Check to make sure the window is big enough

  state.plotwinsize[1] = state.plotwinsize[1] > $
                           state.pixpp*state.norders
  
  plotprofiles_plotupdate

;  Put cursor in window if need user input
  
  if n_elements(GUESSNAP) or n_elements(FIXNAP) then begin

     widget_control, state.plotwin,GET_DRAW_VIEW=view    
     tvcrs,state.plotwinsize[0]/2.,view[1]+state.scrollsize[1]/2.,/DEVICE
     plots, [0.5,0.5],[0,1],COLOR=7,/NORM
     
  endif

end
