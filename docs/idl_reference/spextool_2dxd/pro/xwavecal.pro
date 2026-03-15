pro xwavecal_cleanup,base

  widget_control, base, GET_UVALUE=state, /NO_COPY
  if n_elements(state) ne 0 then begin

     ptr_free, state.x
     ptr_free, state.y
     ptr_free, state.id
     ptr_free, state.lines
     ptr_free, state.types
     ptr_free, state.xpos
     ptr_free, state.w2pcoeffs
     ptr_free, state.p2wcoeffs
     ptr_free, state.slines	
     ptr_free, state.wave
     ptr_free, state.resid

     ptr_free, state.fit
     ptr_free, state.overlay


     ptr_free, state.spec
     ptr_free, state.orders
     
  endif

  state = 0B

end
;
;===============================================================================
;
pro xwavecal_changeorder,state


  z = where(*state.orders eq state.plotorder)
  
  *state.x = (*state.spec)[*,0,z] 
  *state.y = (*state.spec)[*,1,z]
  
  idx = mc_nantrim(*state.x,2)
  
  *state.x = (*state.x)[idx]
  *state.y = (*state.y)[idx]
  
  state.xrange = [min(*state.x,MAX=max),max]
  state.absxrange = state.xrange
  
;  Smooth avoid bad pixels
  
  x = findgen(n_elements(*state.x))
  smooth = mc_robustsg(x,*state.y,5,3,0.1,CANCEL=cancel)
  if cancel then return
  
  min = min(smooth[*,1],/NAN,MAX=max)
  del = (max-min)*0.1
  
  state.yrange = [min-del,max+del]
  state.absyrange = state.yrange

;  Update outfile name

  widget_control, state.oname_fld[1],SET_VALUE='Order'+ $
                  strtrim(string(state.plotorder,FORMAT='(I3)'),2)

  *state.xpos = !values.f_nan
  *state.wave = ''
  state.linereg = !values.f_nan
  *state.fit = !values.f_nan
  *state.overlay = !values.f_nan

  widget_control, state.lines_dl,SET_VALUE='None'
  
end
;
;==============================================================================
;
pro xwavecal_deleteline,state


  idx = findgen(n_elements(*state.xpos))

  z = where(idx ne state.linenum)

  *state.xpos = (*state.xpos)[z]
  *state.wave = (*state.wave)[z]
  *state.id = (*state.id)[z]

  widget_control, state.lines_dl,SET_VALUE=strtrim(*state.wave,2)

  state.linenum = 0
  xwavecal_fitdisp,state
  xwavecal_plotupdate,state
 
  end
;
;==============================================================================
;
pro xwavecal_fitdisp,state

  if n_elements(*state.xpos) lt 2 then return

  fitdeg = (n_elements(*state.xpos)-1) < state.fitdeg
  
  *state.p2wcoeffs = mc_polyfit1d(*state.xpos,double(*state.wave),fitdeg, $
                                  /SILENT,CANCEL=cancel)
  if cancel then return
  
  *state.w2pcoeffs = mc_polyfit1d(double(*state.wave),*state.xpos,fitdeg, $
                                  /SILENT,CANCEL=cancel)
  if cancel then return
    
  *state.resid = *state.xpos-poly(double(*state.wave),*state.w2pcoeffs)

end
;
;===============================================================================
;
pro xwavecal_fitline,state

  z = where(*state.x ge state.linereg[0] and *state.x le state.linereg[1],cnt)
  
  case state.fittype of 
     
     0: begin
        
        tabinv,*state.x,state.linereg[0],lidx
        tabinv,*state.x,state.linereg[1],ridx
        lidx = mc_roundgt(lidx)
        ridx = mc_roundgt(ridx)
        
        *state.fit = double( total((*state.x)[lidx:ridx]* $
                                   (*state.y)[lidx:ridx],/DOUBLE)/$
                             total((*state.y)[lidx:ridx],/DOUBLE) )
        
        *state.overlay = !values.f_nan
        
        widget_control, state.xpos_fld[1],SET_VALUE=strtrim(*state.fit,2)
                
     end
     
     1: begin

        if cnt le 4 then begin
 
           ok = dialog_message(['Not enough points.'],/INFO, $
                               DIALOG_PARENT=state.xwavecal_base)
           return


        endif
        
        result = mpfitpeak(double((*state.x)[z]),double((*state.y)[z]), $
                           a,NTERMS=3)
        *state.r.fit = [a[1],a[2],a[0]]
        *state.p.overlay = [[(*state.x)[z]],[[result]]]
        
        widget_control, state.xpos_fld[1],SET_VALUE=strtrim(a[1],2)
        
     end
     
     2: begin

        if cnt le 5 then begin
 
           ok = dialog_message(['Not enough points.'],/INFO, $
                               DIALOG_PARENT=state.xwavecal_base)
           return


        endif
        
        result = mpfitpeak(double((*state.x)[z]),double((*state.y)[z]), $
                           a,NTERMS=4)
        *state.fit = [a[1],a[2],a[0],a[3]]
        *state.overlay = [[(*state.x)[z]],[[result]]]
        
        widget_control, state.xpos_fld[1],SET_VALUE=strtrim(a[1],2)

     end
     
     3: begin

        if cnt le 6 then begin
 
           ok = dialog_message(['Not enough points.'],/INFO, $
                               DIALOG_PARENT=state.xwavecal_base)
           return


        endif
        
        result = mpfitpeak(double((*state.x)[z]),double((*state.y)[z]), $
                           a,NTERMS=5)
        *state.fit = [a[1],a[2],a[0],a[3],a[4]]
        *state.overlay = [[(*state.x)[z]],[[result]]]
        
        widget_control, state.xpos_fld[1],SET_VALUE=strtrim(a[1],2)
        
     end

     4: begin

        if cnt le 4 then begin
 
           ok = dialog_message(['Not enough points.'],/INFO, $
                               DIALOG_PARENT=state.xwavecal_base)
           return


        endif
        
        result = mpfitpeak(double((*state.x)[z]),double((*state.y)[z]), $
                           a,NTERMS=3,/LORENTZIAN)
        *state.r.fit = [a[1],a[2],a[0]]
        *state.p.overlay = [[(*state.x)[z]],[[result]]]
        
        widget_control, state.xpos_fld[1],SET_VALUE=strtrim(a[1],2)
        
     end

     5: begin

        if cnt le 5 then begin
 
           ok = dialog_message(['Not enough points.'],/INFO, $
                               DIALOG_PARENT=state.xwavecal_base)
           return


        endif
        
        result = mpfitpeak(double((*state.x)[z]),double((*state.y)[z]), $
                           a,NTERMS=4,/LORENTZIAN)
        *state.fit = [a[1],a[2],a[0],a[3]]
        *state.overlay = [[(*state.x)[z]],[[result]]]
        
        widget_control, state.xpos_fld[1],SET_VALUE=strtrim(a[1],2)
        
     end     

     6: begin

        if cnt le 6 then begin
 
           ok = dialog_message(['Not enough points.'],/INFO, $
                               DIALOG_PARENT=state.xwavecal_base)
           return


        endif
        
        result = mpfitpeak(double((*state.x)[z]),double((*state.y)[z]), $
                           a,NTERMS=5,/LORENTZIAN)
        *state.fit = [a[1],a[2],a[0],a[3],a[4]]
        *state.overlay = [[(*state.x)[z]],[[result]]]
        
        widget_control, state.xpos_fld[1],SET_VALUE=strtrim(a[1],2)
        
     end
     
  endcase

  if n_elements(*state.xpos) ge 2 then begin

     lam = poly((*state.fit)[0],*state.p2wcoeffs)
     xmin = min(*state.x,MAX=xmax,/NAN)
     wrange = poly([xmin,xmax],*state.p2wcoeffs)

     z = where(*state.lines gt wrange[0] and *state.lines lt wrange[1],cnt)
     if cnt ne 0 then begin
     
        lines = (*state.lines)[z]
        slines = (*state.slines)[z]
        del = abs(replicate(lam,cnt)-lines)
        
        min = min(del,idx)
        widget_control, state.wave_fld[1],SET_VALUE=strtrim(slines[idx],2)

     endif
        
  endif

  tvcrs,-0.1,0.5,/NORM

  mc_setfocus,state.wave_fld
  
end
;
;==============================================================================
;
pro xwavecal_loadspec,state

;  Get files.

  file = mc_cfld(state.ispectrum_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  file = mc_cfile(file,WIDGET_ID=state.xwavecal_base,CANCEL=cancel)
  if cancel then return

  linefile = mc_cfld(state.ilist_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return

;  Load spectrum

  mc_readspec,file,spc,hdr,obsmode,start,stop,norders,naps,orders,$
              xunits,yunits,slith_pix,slith_arc,slitw_pix,slitw_arc,$
              rp,airmass,xtitle,ytitle,/SILENT,CANCEL=cancel
  if cancel then return

  *state.spec = spc
  *state.orders = orders
  state.norders = norders
  
  widget_control,state.orders_dl,$
                 SET_VALUE=strtrim(string(orders,FORMAT='(I3)'),2) 

;  Get plot info
  
  state.xtitle = xtitle
  state.ytitle = ytitle

;  Get line list

  case state.delimiter of
     
     1: delimiter='|'
     
     2: delimiter='&'
     
     3: delimiter=','
     
     else:
     
  endcase

  case state.columns of
     
     1: begin

        readcol,linefile,wave,COMMENT='#',FORMAT='A',/SILENT
        type = strarr(n_elements(wave))
        
     end

     2: readcol,linefile,wave,type,COMMENT='#',FORMAT='A,A',/SILENT, $
                DELIMITER=delimiter

  endcase

  dwave = double(wave)
  
  s = sort(dwave)
  wave = wave[s]
  type = type[s]
  dwave = dwave[s]
  
  *state.slines = wave
  *state.lines = dwave
  *state.types = type
  *state.xpos = !values.f_nan
  *state.wave = ''

;  Get things going
  
  state.plotorder = orders[0]
  xwavecal_changeorder,state
  xwavecal_plotupdate,state
  state.freeze = 0

end
;
;===============================================================================
;
pro xwavecal_plotspec,state

  mc_mkct
  !x.thick=1
  !p.thick=1
  !y.thick=1
  !p.font=-1
  
  if n_elements(*state.xpos) lt 2 then begin
     
     if state.plottype eq 'Fit Line' then begin
        
        plot,*state.x,*state.y,PSYM=10,/XSTY,/YSTY,XRANGE=state.xrange,$
             YRANGE=state.yrange,XTITLE=state.xtitle,YTITLE=state.ytitle, $
             CHARSIZE=1.5,BACKGROUND=20
        
     endif else begin
        
        xyouts,0.5,0.5,'Not enough points for a fit.',/NORM,CHARSIZE=2, $
               ALIGNMENT=0.5
        return
        
     endelse
     
  endif else begin
     
     x = poly(*state.lines,*state.w2pcoeffs)
     
     nlines = n_elements(*state.lines)
     
     min =min(double(*state.wave),MAX=max)
     
;  Find lower point

     tabinv,*state.lines,min,minidx
     tabinv,*state.lines,max,maxidx

     for i = round(minidx),0,-1 do begin

        minstop = (*state.lines)[i]
        if x[i] lt min(*state.x) or x[i] gt max(*state.x) then break
        
     endfor
     
;  Find upper point
     
     for i = round(maxidx),nlines-1 do begin
        
        maxstop = (*state.lines)[i]
        if x[i] lt min(*state.x) or x[i] gt max(*state.x) then break
        
     endfor

     z = where(*state.lines ge minstop and *state.lines le maxstop,cnt)
     
     if state.plottype eq 'Fit Line' then begin

        lineid_plot,*state.x,*state.y,x[z],strtrim((*state.slines)[z],2)+' '+$
                    strtrim((*state.types)[z],2),/EXTEND,$
                    PSYM=10,/XSTY,/YSTY,XRANGE=state.xrange,$
                    YRANGE=state.yrange,XTITLE=state.xtitle, $
                    YTITLE=state.ytitle,CHARSIZE=1.5,LCHARSIZE=1.5,$
                    /TRADITIONAL,BACKGROUND=20

     endif

     if state.plottype eq 'Solution' then begin
        
        lineid_plot,*state.xpos,double(*state.wave),x[z], $
                    strtrim((*state.lines)[z],2)+' '+$
                    strtrim((*state.types)[z],2), $
                    /EXTEND,PSYM=1,/XSTY,/YSTY,XRANGE=state.xrange,$
                    XTITLE=state.xtitle,YTITLE='Residual (pixels)', $
                    CHARSIZE=1.5,LCHARSIZE=1.5,/TRADITIONAL,BACKGROUND=20

        plotsym,0,1,/FILL
        oplot,*state.xpos,double(*state.wave),COLOR=2,PSYM=8
        xmin = min(*state.xpos,MAX=xmax)
        x = findgen(xmax-xmin+1)+xmin
        oplot,x,poly(x,*state.p2wcoeffs)
        

        xyouts,0.01,0.95,strtrim(n_elements(*state.xpos),2)+' points', $
               COLOR=5,ALIGNMENT=0,/NORM,CHARSIZE=2

        fitdeg = (n_elements(*state.xpos)-1) < state.fitdeg
        xyouts,0.01,0.9,strtrim(fitdeg,2)+' deg', $
               COLOR=5,ALIGNMENT=0,/NORM,CHARSIZE=2

     endif
     
     if state.plottype eq 'Residuals' then begin
        
        lineid_plot,*state.xpos,*state.resid,x[z], $
                    strtrim((*state.lines)[z],2)+' '+$
                    strtrim((*state.types)[z],2), $
                    /EXTEND,PSYM=1,/XSTY,/YSTY,XRANGE=state.xrange,$
                    XTITLE=state.xtitle,YTITLE='Residual (pixels)', $
                    CHARSIZE=1.5,LCHARSIZE=1.5,/TRADITIONAL,BACKGROUND=20

        plotsym,0,1,/FILL
        oplot,*state.xpos,*state.resid,COLOR=2,PSYM=-8
        plots,!x.crange,[0,0],LINESTYLE=1

        xyouts,0.01,0.95,strtrim(n_elements(*state.xpos),2)+' points', $
               COLOR=5,ALIGNMENT=0,/NORM,CHARSIZE=2

        fitdeg = (n_elements(*state.xpos)-1) < state.fitdeg
        xyouts,0.01,0.9,strtrim(fitdeg,2)+' deg', $
               COLOR=5,ALIGNMENT=0,/NORM,CHARSIZE=2

     endif

          
  endelse

  state.pscale = !p
  state.xscale = !x
  state.yscale = !y

; Plot found lines

  for i = 0,n_elements(*state.xpos)-1 do begin

     if (*state.xpos)[i] lt !x.crange[0] or $
        (*state.xpos)[i] gt !x.crange[1] then continue
     
     plots,[(*state.xpos)[i],(*state.xpos)[i]],!y.crange, $
           COLOR=3,LINESTYLE=0

  endfor
     
  if state.plottype eq 'Fit Line' then begin
        
;  Plot Line Region if necessary
     
     plots, [state.linereg[0],state.linereg[0]],!y.crange,COLOR=7, $
            LINESTYLE=2
     plots, [state.linereg[1],state.linereg[1]],!y.crange,COLOR=7, $
            LINESTYLE=2
     
;  Plot fitted line center
     
     plots,[(*state.fit)[0],(*state.fit)[0]],!y.crange,COLOR=4
     
     if n_elements(*state.overlay) gt 1 then begin
        
        oplot,(*state.overlay)[*,0],(*state.overlay)[*,1],COLOR=3,PSYM=10
        
     endif
     
  endif
  
end
;
;===============================================================================
;
pro xwavecal_plotupdate,state

  mc_mkct
  wset, state.pixmap_wid  
  erase,COLOR=20
  xwavecal_plotspec,state

  wset, state.plotwin_wid
  device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,state.pixmap_wid]

end
;
;=============================================================================
;
pro xwavecal_setslider,state

;  Get new slider value
  
  del = state.absxrange[1]-state.absxrange[0]
  midwave = (state.xrange[1]+state.xrange[0])/2.
  state.sliderval = round((midwave-state.absxrange[0])/del*100)
     
  widget_control, state.slider, SET_VALUE=state.sliderval
  
end
;
;===============================================================================
;
pro xwavecal_storeline,state

  xpos = mc_cfld(state.xpos_fld,5,/EMPTY,CANCEL=cancel)
  if cancel then return
  wave = mc_cfld(state.wave_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return

;  Check to see if the line is in the list

  zline = where(strtrim(wave,2) eq strtrim(*state.slines,2),cnt)
  if cnt eq 0 then begin

     ok = dialog_message(['Line does not appear in your line list.'],/INFO, $
                         DIALOG_PARENT=state.xwavecal_base)
     tvcrs,-0.1,0.5,/NORM
     mc_setfocus,state.wave_fld
     return
     

  endif
  
  z = where(finite(*state.xpos) eq 1,cnt)
  if cnt eq 0 then begin

     (*state.xpos) = double(xpos)
     (*state.wave) = wave
     (*state.id) = strjoin((*state.types)[zline],'/')

  endif else begin

     *state.xpos = [*state.xpos,double(xpos)]
     *state.wave = [*state.wave,wave]
     (*state.id) = [*state.id,strjoin((*state.types)[zline],'/')]
     
  endelse

  z = where(finite(*state.xpos) eq 1)
  
  (*state.xpos) = (*state.xpos)[z]
  (*state.wave) = (*state.wave)[z]
  (*state.id) = (*state.id)[z]

  s = sort(*state.xpos)
  
  (*state.xpos) = (*state.xpos)[s]
  (*state.wave) = (*state.wave)[s]
  (*state.id) = (*state.id)[s]

  widget_control, state.wave_fld[1],SET_VALUE=''
  widget_control, state.xpos_fld[1],SET_VALUE=''

  state.linereg = !values.f_nan
  *state.fit = !values.f_nan
  *state.overlay = !values.f_nan
  state.cursormode = 'None'

  xwavecal_fitdisp,state
  xwavecal_plotupdate,state

  widget_control, state.lines_dl,SET_VALUE=strtrim(*state.wave,2)
  state.linenum = 0
  
  widget_control, state.plotwin, /INPUT_FOCUS
  tvcrs,0.5,0.5,/NORM
  
end
;
;===============================================================================
;
pro xwavecal_writefile,state


  ofile = mc_cfld(state.oname_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  ifile = mc_cfld(state.ispectrum_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  linefile = mc_cfld(state.ilist_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return


  
  openw, lun, ofile+'.dat',/GET_LUN,WIDTH=200
  
  printf, lun, mc_datetag()
  printf, lun, '# Spectrum File= ',strtrim(file_basename(ifile))
  printf, lun, '# Line List = ',strtrim(file_basename(linefile))
  printf, lun, '#'
    
  if n_elements(*state.wave) gt 1 then begin

     mc_moments,*state.resid,mean,var,stddev,/SILENT
     
     printf, lun, '# W2P Coeffs= ',*state.w2pcoeffs
     printf, lun, '# P2W Coeffs= ',*state.p2wcoeffs
     printf, lun, '#        RMS= ',stddev
     printf, lun, '#'

  endif
     
  mc_filltable,lun,2,$
               {l:'Wavelength',v:*state.wave,f:'A10',u:'(um)'},$
               {l:'X',v:*state.xpos,f:'D12.7',u:'(pixels)'},$
               {l:'Species',v:*state.id,f:'A15',u:''},$
               DELIMITER='|'
               
  
  free_lun, lun

end
;
;===============================================================================
;
pro xwavecal_zoom,state,IN=in,OUT=out

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

    xwavecal_plotupdate,state
    xwavecal_setslider,state

    
end
;
;===============================================================================
;
pro xwavecal_event,event

  widget_control, event.id,  GET_UVALUE = uvalue
  
  if uvalue eq 'Quit' then begin
     
     widget_control, event.top, /DESTROY
     goto, getout
     
  endif

  widget_control, event.top, GET_UVALUE = state, /NO_COPY
  
  case uvalue of

     'Columns': state.columns = event.index+1

     'Delete Line': begin

        if state.freeze then goto, cont
        xwavecal_deleteline,state

     end
     
     'Delimiter': state.delimiter = event.index

     'Fit Degree': begin

        state.fitdeg = event.index
        if state.freeze then goto, cont
        xwavecal_fitdisp,state
        xwavecal_plotupdate,state

     end
     
     'Fit Type': begin

        state.fittype = event.index
        if state.freeze then goto, cont
        state.linereg = !values.f_nan
        *state.fit = !values.f_nan
        *state.overlay = !values.f_nan
        xwavecal_plotupdate,state
        
     end
     
     'Input Line List': begin
        
        obj = dialog_pickfile(DIALOG_PARENT=state.xwavecal_base,$
                              /MUST_EXIST)
        if obj eq '' then goto, cont
        widget_control,state.ilist_fld[1],SET_VALUE =strtrim(obj,2)
        mc_setfocus,state.ilist_fld
        
     end
     
     'Input Spectrum': begin
        
        obj = dialog_pickfile(DIALOG_PARENT=state.xwavecal_base,$
                              /MUST_EXIST,FILTER='*.fits')
        if obj eq '' then goto, cont
        widget_control,state.ispectrum_fld[1],SET_VALUE =strtrim(obj,2)
        mc_setfocus,state.ispectrum_fld
        
     end
     
     'Load Spectrum': xwavecal_loadspec,state

     'Plot Order': begin

        if state.freeze then goto, cont
        state.plotorder = (*state.orders)[event.index]
        xwavecal_changeorder,state
        xwavecal_plotupdate,state

     end
     
     'Plot Type': begin

        if state.freeze then goto, cont
        state.plottype = event.value
        xwavecal_plotupdate,state
     
     end

     'Stored Lines': begin

        if state.freeze then goto, cont
        state.linenum = event.index

     end
     
     'Store Position': begin

        if state.freeze then goto, cont
        xwavecal_storeline, state

     end
     
     'Wavelength Field': begin

        if state.freeze then goto, cont
        xwavecal_storeline, state

     end
        
     'Write File': begin
        
        if state.freeze then goto, cont
        xwavecal_writefile,state
        
     end
     
     else:

  endcase

cont: 

  widget_control, state.xwavecal_base, SET_UVALUE=state, /NO_COPY
  
getout:

end
;
;===============================================================================
;
pro xwavecal_plotwinevent,event

  widget_control, event.top, GET_UVALUE = state, /NO_COPY
  widget_control, event.id,  GET_UVALUE = uvalue

  if state.plottype eq 'Residuals' then goto, cont

;  Check to see if it is a TRACKING event.

  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then $
     begin

     widget_control, state.plotwin,INPUT_FOCUS=event.enter
     
     wset, state.plotwin_wid
     device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,$
                   state.pixmap_wid]
     goto, cont
     
  endif

  if event.type eq 6 and event.release eq 0 then begin

     case event.key of

        5: begin

           state.sliderval = (state.sliderval-1) > 0
           del = state.absxrange[1]-state.absxrange[0]
           oldcen = (state.xrange[1]+state.xrange[0])/2.
           newcen = state.absxrange[0]+del*(state.sliderval/100.)
           state.xrange = state.xrange + (newcen-oldcen)
           xwavecal_setslider,state
           xwavecal_plotupdate,state

           
        end

        6: begin

           state.sliderval = (state.sliderval+1) < 100
           del = state.absxrange[1]-state.absxrange[0]
           oldcen = (state.xrange[1]+state.xrange[0])/2.
           newcen = state.absxrange[0]+del*(state.sliderval/100.)
           state.xrange = state.xrange + (newcen-oldcen)
           xwavecal_setslider,state
           xwavecal_plotupdate,state
           
        end
           
        else:
           
     endcase
        
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
           state.linereg = !values.f_nan
           *state.fit = !values.f_nan
           *state.overlay = !values.f_nan
           xwavecal_plotupdate,state
           
        end

        'i': xwavecal_zoom,state,/IN

        'o': xwavecal_zoom,state,/OUT

        's': begin
           
           state.cursormode = 'Select'
           state.linereg = !values.f_nan
           *state.fit = !values.f_nan
           *state.overlay = !values.f_nan
           xwavecal_plotupdate,state
           
           end
        
        'w': begin
           
           state.xrange = state.absxrange
           state.yrange = state.absyrange
           xwavecal_setslider,state
           xwavecal_plotupdate,state
           
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

  
;  If not, set the keyboard focus and active window.

  !p = state.pscale
  !x = state.xscale
  !y = state.yscale
  x  = event.x/float(state.plot_size[0])
  y  = event.y/float(state.plot_size[1])
  xy = convert_coord(x,y,/NORMAL,/TO_DATA)

  if event.type eq 1 then begin

     case state.cursormode of 

        'Select': begin
           
           if event.type ne 1 then goto, cont     
           z = where(finite(state.linereg) eq 1,count)        

           if count eq 0 then begin
              
              state.linereg[0] = xy[0]
              xwavecal_plotupdate,state
              
           endif

           if count eq 1 then begin
              
              state.linereg[1] = xy[0]
              state.linereg = state.linereg[sort(state.linereg)]
              xwavecal_fitline,state
              xwavecal_plotupdate,state
              
           endif
           
        end
        
        'Zoom': begin
           
           z = where(finite(state.reg) eq 1,count)
           if count eq 0 then state.reg[*,0] = xy[0:1] else begin 
              
              state.reg[*,1] = xy[0:1]
              state.xrange   = [min(state.reg[0,*],MAX=max),max]
              state.yrange   = [min(state.reg[1,*],MAX=max),max]
              xwavecal_setslider,state
              xwavecal_plotupdate,state
              state.cursormode = 'None'
              state.reg = !values.f_nan
              
           endelse

        end

        'XZoom': begin

           z = where(finite(state.reg) eq 1,count)
           if count eq 0 then begin
              
              state.reg[*,0] = xy[0:1]
              wset, state.pixmap_wid
              plots, [event.x,event.x],[0,state.plot_size[1]],COLOR=2,$
                     /DEVICE,LINESTYLE=2
              wset, state.plotwin_wid
              device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,$
                            0,state.pixmap_wid]
              
           endif else begin

              state.reg[*,1] = xy[0:1]
              state.xrange = [min(state.reg[0,*],max=m),m]
              state.cursormode = 'None'
              state.reg = !values.f_nan
              xwavecal_setslider,state
              xwavecal_plotupdate,state
              
           endelse


        end

        'YZoom': begin

           z = where(finite(state.reg) eq 1,count)
           if count eq 0 then begin
              
              state.reg[*,0] = xy[0:1]
              wset, state.pixmap_wid
              plots, [0,state.plot_size[0]],[event.y,event.y],COLOR=2,$
                     /DEVICE,LINESTYLE=2

              wset, state.plotwin_wid
              device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,$
                            0,state.pixmap_wid]
              
           endif else begin

              state.reg[*,1] = xy[0:1]
              state.yrange = [min(state.reg[1,*],max=m),m]
              state.cursormode = 'None'
              state.reg[*] = !values.f_nan
              xwavecal_setslider,state
              xwavecal_plotupdate,state
              
           endelse

        end

        else:

     endcase

     
  endif

;  Copy the pixmaps and draw the lines.

  wset, state.plotwin_wid
  device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,state.pixmap_wid]

  case state.cursormode of 

     'XZoom': begin

        wset, state.plotwin_wid
        plots, [event.x,event.x],[0,state.plot_size[1]],COLOR=2,/DEVICE

     end
     'YZoom': plots, [0,state.plot_size[0]],[event.y,event.y],COLOR=2,/DEVICE

     'Zoom': begin

        plots, [event.x,event.x],[0,state.plot_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot_size[0]],[event.y,event.y],COLOR=2,/DEVICE
        xy = convert_coord(event.x,event.y,/DEVICE,/TO_DATA)
        plots,[state.reg[0,0],state.reg[0,0]],[state.reg[1,0],xy[1]],$
              LINESTYLE=2,COLOR=2
        plots, [state.reg[0,0],xy[0]],[state.reg[1,0],state.reg[1,0]],$
               LINESTYLE=2,COLOR=2
        
     end

     else: begin

        plots, [event.x,event.x],[0,state.plot_size[1]],COLOR=2,/DEVICE
        plots, [0,state.plot_size[0]],[event.y,event.y],COLOR=2,/DEVICE

     end

  endcase

;  Update cursor position.
  
  if not state.freeze then begin

     tabinv, *state.x,xy[0],idx
     idx = round(idx)
     label = 'Cursor X: '+strtrim(xy[0],2)+', Y:'+strtrim(xy[1],2)
     label = label+'   Spectrum X: '+strtrim( (*state.x)[idx],2)+$
             ', Y:'+strtrim( (*state.y)[idx],2)
     widget_control,state.message,SET_VALUE=label

  endif

cont:
  
  widget_control, state.xwavecal_base, SET_UVALUE=state, /NO_COPY
  
end
;
;===============================================================================
;
pro xwavecal_resizeevent,event

  widget_control, event.top, GET_UVALUE = state, /NO_COPY
  
  if n_params() eq 0 then begin
     
     size = widget_info(state.xwavecal_base, /GEOMETRY)
     xsize = size.xsize
     ysize = size.ysize
     
  endif else begin
     
     widget_control, state.xwavecal_base, TLB_GET_SIZE=size
     xsize = size[0]
     ysize = size[1]
     
  endelse

  state.plot_size[0] = xsize-state.buffer[0]
  state.plot_size[1] = ysize-state.buffer[1]
  
  widget_control, state.xwavecal_base,UPDATE=0
  widget_control, state.plotwin, DRAW_XSIZE=state.plot_size[0], $
                  DRAW_YSIZE=state.plot_size[1]
  widget_control, state.xwavecal_base,UPDATE=1
  
  wdelete,state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plot_size[0],YSIZE=state.plot_size[1]
  state.pixmap_wid = !d.window

  if state.freeze then begin

     erase, COLOR=20
     wset, state.plotwin_wid
     device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,$
                   state.pixmap_wid]
     
  endif else xwavecal_plotupdate,state

  widget_control, state.xwavecal_base, SET_UVALUE=state, /NO_COPY
  
end  
;
;===============================================================================
;
pro xwavecal
  
  mc_mkct
  device, RETAIN=2

  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

;  Get screen size

  screensize = get_screen_size()
 
  state = {absxrange:[0.,0.],$
           absyrange:[0.,0.],$
           buffer:[0L,0L],$
           columns:2,$
           cursormode:'None',$
           delimiter:1,$
           fit:ptr_new(2),$
           fitdeg:1,$
           fittype:2,$
           freeze:1,$
           id:ptr_new(''),$
           ilist_fld:[0L,0L],$
           ispectrum_fld:[0L,0L],$
           lines:ptr_new(2),$
           linenum:0L,$
           lines_dl:0L,$
           linereg:[!values.f_nan,!values.f_nan],$
           message:0L,$
           norders:0,$
           oname_fld:[0L,0L],$
           orders:ptr_new(2),$
           orders_dl:0L,$
           overlay:ptr_new(2),$           
           pixmap_wid:0L,$
           p2wcoeffs:ptr_new(2),$
           plotorder:0,$
           plot_size:[screensize[0]*0.45,screensize[1]*0.5 > 550],$
           plottype:'Fit Line',$
           plotwin:0L,$
           plotwin_wid:0L,$           
           pscale:!p,$
           reg:[[!values.d_nan,!values.d_nan],$
                [!values.d_nan,!values.d_nan]],$
           resid:ptr_new(2),$
           slider:0L,$
           sliderval:50,$
           slines:ptr_new(''),$
           spec:ptr_new(2),$
           types:ptr_new(2),$
           x:ptr_new(2),$
           xpos:ptr_new(2),$
           xpos_fld:[0L,0L],$
           xrange:[0.,0.],$
           xscale:!x,$
           xtitle:'',$
           y:ptr_new(2),$
           yrange:[0.,0.],$
           yscale:!y,$
           ytitle:'',$
           w2pcoeffs:ptr_new(2),$
           wave:ptr_new(2),$
           wave_fld:[0L,0L],$
           xwavecal_base:0L}
  
  state.xwavecal_base = widget_base(TITLE='Xwavecal', $
                                    /COLUMN,$
                                    /TLB_SIZE_EVENTS)

     button = widget_button(state.xwavecal_base,$
                            FONT=buttonfont,$
                            EVENT_PRO='xwavecal_event',$
                            VALUE='Quit',$
                            UVALUE='Quit')

     row_base = widget_base(state.xwavecal_base,$
                            /ROW)

        col1_base = widget_base(row_base,$
                                EVENT_PRO='xwavecal_event',$
                                /COLUMN)
        
           box1_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box1_base,$
                                   VALUE='1.  Load Spectra',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 input = widget_button(row,$
                                       FONT=buttonfont,$
                                       VALUE='Input Spectrum',$
                                       UVALUE='Input Spectrum')
                 
                 input_fld = coyote_field2(row,$
                                           LABELFONT=buttonfont,$
                                           FIELDFONT=textfont,$
                                           TITLE=':',$
                                           VALUE='cspectra334-341.fits',$
                                           UVALUE='Input Spectrum Field',$
                                           XSIZE=15,$
                                           TEXTID=textid)
                 state.ispectrum_fld = [input_fld,textid]  

              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 input = widget_button(row,$
                                       FONT=buttonfont,$
                                       VALUE='Input Line List',$
                                       UVALUE='Input Line List')
                 
                 input_fld = coyote_field2(row,$
                                           LABELFONT=buttonfont,$
                                           FIELDFONT=textfont,$
                                           TITLE=':',$
                                           VALUE='pnlines.dat',$
                                           UVALUE='Input Line List Field',$
                                           XSIZE=15,$
                                           TEXTID=textid)
                 state.ilist_fld = [input_fld,textid]

                 
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)

                 label = widget_label(row,$
                                      VALUE='Columns:',$
                                      FONT=buttonfont,$
                                      /ALIGN_LEFT)
                 
                 cbox = widget_combobox(row,$
                                        FONT=buttonfont,$
                                        VALUE=['Wavelengths','Wavelengths, IDs'],$
                                        SCR_XSIZE=140,$
                                        UVALUE='Columns')
                 widget_control, cbox,SET_COMBOBOX_SELECT=state.columns-1

                 
                 row = widget_base(box1_base,$
                                   /ROW,$
                                   /BASE_ALIGN_CENTER)
                 
                 label = widget_label(row,$
                                      VALUE='Delimiter:',$
                                      FONT=buttonfont,$
                                      /ALIGN_LEFT)
                 
                 cbox = widget_combobox(row,$
                                        FONT=buttonfont,$
                                        VALUE=['None',' |',' &',' ,'],$
                                        SCR_XSIZE=70,$
                                        UVALUE='Delimiter')
                 widget_control, cbox,SET_COMBOBOX_SELECT=state.delimiter

              load = widget_button(box1_base,$
                                   VALUE='Load Spectrum',$
                                   UVALUE='Load Spectrum',$
                                   FONT=buttonfont)

           box2_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box2_base,$
                                   VALUE='2.  Identify Lines',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              state.orders_dl = widget_droplist(box2_base,$
                                                FONT=buttonfont,$
                                                TITLE='Plot Order: ',$
                                                VALUE='01',$
                                                UVALUE='Plot Order')

              input_fld = coyote_field2(box2_base,$
                                        LABELFONT=buttonfont,$
                                        FIELDFONT=textfont,$
                                        TITLE='Wavelength:',$
                                        UVALUE='Wavelength Field',$
                                        XSIZE=15,$
                                        /CR_ONLY,$
                                        EVENT_PRO='xwavecal_event',$
                                        TEXTID=textid)
              state.wave_fld = [input_fld,textid]

              input_fld = coyote_field2(box2_base,$
                                        LABELFONT=buttonfont,$
                                        FIELDFONT=textfont,$
                                        TITLE='X Pos:',$
                                        UVALUE='X Pos Field',$
                                        XSIZE=15,$
                                        TEXTID=textid)
              state.xpos_fld = [input_fld,textid]

              button = widget_button(box2_base,$
                                     VALUE='Store Position',$
                                     UVALUE='Store Position',$
                                     FONT=buttonfont)

           box3_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)

              label = widget_label(box3_base,$
                                   VALUE='3.  Adjust Fit',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)

                 dl = widget_droplist(box3_base,$
                                      VALUE=['0','1','2','3','4','5','6', $
                                             '7','8','9'],$
                                      UVALUE='Fit Degree',$
                                      TITLE='Fit Degree: ',$
                                      FONT=buttonfont)
                 widget_control, dl,SET_DROPLIST_SELECT=1


                 row = widget_base(box3_base,$
                                   /ROW,$
                                   /BASE_ALIGN_CENTER)

                    state.lines_dl = widget_droplist(row,$
                                                     VALUE=['None'],$
                                                     UVALUE='Stored Lines',$
                                                     TITLE='Stored Lines: ',$
                                                     /DYNAMIC_RESIZE,$
                                                     FONT=buttonfont)

                    button = widget_button(row,$
                                           VALUE='Delete',$
                                           UVALUE='Delete Line',$
                                           FONT=buttonfont)
                                     
           box4_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)
           
              label = widget_label(box4_base,$
                                   VALUE='4.  Write Spectra to File',$
                                   FONT=buttonfont,$
                                   /ALIGN_LEFT)
              
              oname = coyote_field2(box4_base,$
                                    LABELFONT=buttonfont,$
                                    FIELDFONT=textfont,$
                                    TITLE='File Name:',$
                                    UVALUE='Object File Oname',$
                                    XSIZE=18,$
                                    TEXTID=textid)
              state.oname_fld = [oname,textid]
              
              write = widget_button(box4_base,$
                                    VALUE='Write File',$
                                    UVALUE='Write File',$
                                    FONT=buttonfont)
                    
        col2_base = widget_base(row_base,$
                                EVENT_PRO='xwavecal_event',$
                                /COLUMN)
        
           state.message = widget_text(col2_base, $
                                       YSIZE=1)

           row = widget_base(col2_base,$
                             /ROW,$
                            /BASE_ALIGN_CENTER)


           types =['Centroid','Gaussian','Gaussian+Constant','Gaussian+Line',$
                  'Lorentzian','Lorentzian+Constant','Lorentzian+Line']
              dl = widget_droplist(row,$
                                   VALUE=types,$
                                   UVALUE='Fit Type',$
                                   TITLE='Fit Type: ',$
                                   FONT=buttonfont)
              widget_control, dl, SET_DROPLIST_SELECT=state.fittype

              bf = cw_bgroup(row,$
                             FONT=buttonfont,$
                             LABEL_LEFT='Plot Type: ',$
                             ['Fit Line','Solution','Residuals'],$
                             /EXCLUSIVE,$
                             /ROW,$
                             /RETURN_NAME,$
                             /NO_RELEASE,$
                             SET_VALUE=0,$
                             UVALUE='Plot Type')

              state.plotwin = widget_draw(col2_base,$
                                          /ALIGN_CENTER,$
                                          XSIZE=state.plot_size[0],$
                                          YSIZE=state.plot_size[1],$
                                          EVENT_PRO='xwavecal_plotwinevent',$
                                          /KEYBOARD_EVENTS,$
                                          /BUTTON_EVENTS,$
                                          /TRACKING_EVENTS,$
                                          /MOTION_EVENTS)
              
           state.slider = widget_slider(col2_base,$
                                        UVALUE='Slider',$
                                        EVENT_PRO='xwavecal_event',$
                                        /DRAG,$
                                        /SUPPRESS_VALUE,$
                                        FONT=buttonfont)
           widget_control, state.slider, SET_VALUE=state.sliderval

; Get things running.  Center the widget using the Fanning routine.
           
  cgcentertlb,state.xwavecal_base
  widget_control, state.xwavecal_base, /REALIZE

;  Get plotwin ids
  
  widget_control, state.plotwin, GET_VALUE=x
  state.plotwin_wid = x

;  Create pixmap windows

  window, /FREE, /PIXMAP,XSIZE=state.plot_size[0],YSIZE=state.plot_size[1]
  state.pixmap_wid = !d.window
  
  erase, COLOR=20
  wset, state.plotwin_wid
   device, COPY=[0,0,state.plot_size[0],state.plot_size[1],0,0,$
                 state.pixmap_wid]
  
;  Get sizes for things.

  widget_geom = widget_info(state.xwavecal_base, /GEOMETRY)
  state.buffer[0]=widget_geom.xsize-state.plot_size[0]
  state.buffer[1]=widget_geom.ysize-state.plot_size[1]
  
; Start the Event Loop. This will be a non-blocking program.

  XManager, 'xwavecal', $
            state.xwavecal_base, $
            CLEANUP='xwavecal_cleanup',$
            EVENT_HANDLER='xwavecal_resizeevent',$
            /NO_BLOCK
  
; Put state variable into the user value of the top level base.
  
  widget_control, state.xwavecal_base, SET_UVALUE=state, /NO_COPY




end
