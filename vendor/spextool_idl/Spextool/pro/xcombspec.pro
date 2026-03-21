;+
; NAME:
;     xcombspec
;    
; PURPOSE:
;     Combines SpeX spectra.
;    
; CATEGORY:
;     Widget
;
; CALLING SEQUENCE:
;     xcombspec
;    
; INPUTS:
;     None
;    
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     None
;     
; OUTPUTS:
;     Writes a SpeX spectra FITS file to disk
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     See xcombspec_helpfile.txt in Spextool/helpfiles
;
; EXAMPLE:
;     
; MODIFICATION HISTORY:
;     2000       - Written by M. Cushing, Institute for Astronomy, UH
;     2002-07-19 - Heavily modified by M. Cushing
;     2005-04-x  - Added new plotting controls.
;     2005-10-20 - Modified so that NaNs in the spectra are not
;                  removed before the interpolation is performed in
;                  the loadimages program
;     2008-02-13 - Removed output as a text file as an option.
;     2013       - Many modifications
;     2017       - Major modifications for iSHELL
;     2019-03-22 - Added a shift routine specific to NIHTS
;-
;
;===============================================================================
;
; ------------------------------Event Handlers-------------------------------- 
;
;===============================================================================
;
pro xcombspec_event, event

  widget_control, event.id,  GET_UVALUE = uvalue
  if uvalue eq 'Quit' then begin
          
     widget_control, event.top, /DESTROY
     goto, getout
     
  endif
  
  widget_control, event.top, GET_UVALUE = state, /NO_COPY
  widget_control, /HOURGLASS
  
  case uvalue of
     
     'Aperture': begin
        
        if state.freeze then goto, cont
        state.ap = event.index
        
        z = where((*state.orders) eq (*state.scaleorder)[state.ap])
        widget_control, state.scaleorder_dl, SET_DROPLIST_SELECT=total(z)+1
        xcombspec_plotupdate,state
        
     end
     
     'Write File': begin
        
        if state.freeze then goto, cont
        xcombspec_writefile,state
        
     end

     'Combination Statistic':  begin
        
        state.combinestat = event.index
        sensitive = (event.index le 2) ? 1:0
        widget_control, state.rthresh_fld[0], SENSITIVE=sensitive
        if state.freeze then goto, cont
        xcombspec_combinespec,state,CANCEL=cancel
        if cancel then goto, cont
        xcombspec_plotupdate,state
        
     end

     'Correct Spectral Shape': xcombspec_correctspec,state
     
     'File Name': begin
        
        path =dialog_pickfile(DIALOG_PARENT=state.xcombspec_base,$
                              /MUST_EXIST)
        if path ne '' then widget_control,state.filename_fld[1],SET_VALUE=path
        
     end
     
     'Help': begin

        pre = (strupcase(!version.os_family) eq 'WINDOWS') ? 'start ':'open '
        
        spawn, pre+filepath(strlowcase(state.instrument)+'_spextoolmanual.pdf',$
                            ROOT=state.packagepath,SUBDIR='manual')
        
     end
     
     'Input Prefix': mc_setfocus, state.cfiles_fld
     
     'Load Spectra': xcombspec_loadspec,state

     'Plot Order': begin

        if state.plotwinsize[1] eq state.scrollsize[1] then goto, cont

        del = max(*state.orders,MIN=min)-min+1
        offset = state.plotwinsize[1]/float((del+1))
        frac = ((reverse(*state.orders))[event.index]-min)/float(del)
        
        widget_control, state.plotwin, $
                        SET_DRAW_VIEW=[0,state.plotwinsize[1]*frac-offset]
        
     end
     
     'Path Button': begin
        
        path= dialog_pickfile(/DIRECTOR,DIALOG_PARENT=state.xcombspec_base,$
                              TITLE='Select Path',/MUST_EXIST)
        
        if path ne '' then begin
           
           path = mc_cpath(path,WIDGET_ID=state.xcombspec_base,CANCEL=cancel)
           if cancel then return
           widget_control,state.path_fld[1],SET_VALUE = path
           mc_setfocus,state.path_fld
           
        endif
        
     end
     
     'Plot Type': begin
        
        state.plottype = event.index
        if state.freeze then goto, cont
        xcombspec_plotupdate,state
        
     end

     'Prune Order': begin

        if state.freeze then goto, cont
        state.pruneorderidx = state.norders-1-event.index
        
     end

     'Prune Spectra': begin

        if state.freeze then goto, cont
        xcombspec_prunespec,state
        
     end

     'Readmode': begin
        
        widget_control, state.inprefix_fld[0],  SENSITIVE=0
        if event.value eq 'Filename' then begin
           
           state.filereadmode = event.value
           widget_control, state.inprefix_fld[0], SENSITIVE=0
           mc_setfocus,state.cfiles_fld
           
        endif else begin
           
           state.filereadmode = event.value
           widget_control, state.inprefix_fld[0], /SENSITIVE
           mc_setfocus,state.inprefix_fld
           
        endelse
        
     end
     
     'Robust Threshold': begin
        
        xcombspec_combinespec,state,CANCEL=cancel
        if cancel then goto, cont
        xcombspec_plotupdate,state
        
     end
    
     'Scale Order': begin

        if state.freeze then goto, cont
        state.scaleorderidx = state.norders-1-event.index

     end

     'Scale Spectra': begin

        if state.freeze then goto, cont
        xcombspec_scalespec,state

     end

     'Shift Spectra': begin

        if state.freeze then goto, cont
        xcombspec_shiftspec,state

     end
     
     'Spectra Files Button': begin

        if state.filereadmode eq 'Index' then begin

           message = [['It looks like you are trying to load ' + $
                       'data via a file name.'],$
                      ['Please chnage the File Read Mode to "Filename" first.']]
     
           ok = dialog_message(message,/ERROR, $
                               DIALOG_PARENT=state.xcombspec_base)

           goto, cont
           
        endif 

        
        path = mc_cfld(state.path_fld,7,CANCEL=cancel)
        if cancel then return
        
        fullpath = dialog_pickfile(DIALOG_PARENT=state.xcombspec_base,$
                                   PATH=path,/MUST_EXIST, $
                                   FILTER='*.fits',/MULTIPLE_FILES)
        
        case (size(fullpath))[1] of 
           
           1: begin
              
              if fullpath[0] ne '' then begin
                 
                 widget_control,state.cfiles_fld[1],$
                                SET_VALUE=strmid(fullpath[0],$
                                            strpos(fullpath,'/',/REVERSE_S)+1)
                 mc_setfocus, state.outfile_fld
                 
              endif
              
           end
           
           else: begin
              
              for i =0,(size(fullpath))[1]-1 do begin
                 
                 tmp = strmid(fullpath[i],strpos(fullpath[i],'/',$
                                                 /REVERSE_S)+1)
                 arr = (i eq 0) ? tmp:[arr,tmp]
                 
              endfor
              widget_control,state.cfiles_fld[1],$
                             SET_VALUE=strjoin(arr,',',/SINGLE)
              
           end
           
        endcase
        
     end
     
     'Spectra Files Field': mc_setfocus,state.outfile_fld
     
     'Spectra Type': begin
        
        state.spectype = event.index
        if state.freeze then goto, cont
        xcombspec_plotupdate,state
        
     end
     
     else:
     
  endcase

;  Put state variable into the user value of the top level base.
 
cont: 

  widget_control, state.xcombspec_base, SET_UVALUE=state, /NO_COPY

getout:
  
end
;
;===============================================================================
;
pro xcombspec_plotwinevent,event

  widget_control, event.top, GET_UVALUE = state, /NO_COPY

;  Check tracking
  
  if strtrim(tag_names(event,/STRUCTURE_NAME),2) eq 'WIDGET_TRACKING' then begin
     
     wset, state.plotwin_wid
     device, COPY=[0,0,state.plotwinsize[0],state.plotwinsize[1],0,0, $
                   state.pixmap_wid]
     widget_control, state.plotwin,INPUT_FOCUS=event.enter

     goto, cont

  endif
  
  if state.plottype eq 1 and event.release then begin
     
     idx = floor(event.y/float(state.plotwinsize[1])*state.norders)
     
     case state.spectype of 
        
        0: spec = reform((*state.combspec)[*,1,idx])
        
        1: spec = reform((*state.combspec)[*,2,idx])
        
        2: spec = reform((*state.combspec)[*,1,idx]) / $
                  reform((*state.combspec)[*,2,idx])
        
     endcase
     
     xzoomplot,reform((*state.combspec)[*,0,idx]),spec
     
  endif else begin

;  Check for arrow keys

     if event.type eq 6 and event.press eq 1 then begin
        
        case event.key of
           
           7: begin             ;  up arrow
              
              widget_control, state.plotwin,GET_DRAW_VIEW=current
              offset = state.plotwinsize[1]/state.norders
              max = state.plotwinsize[1]-state.scrollsize[1]
              val = (current[1]+offset) < max
              widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
              
           end
           
           8: begin             ;  down arrow
              
              widget_control, state.plotwin,GET_DRAW_VIEW=current
              offset = state.plotwinsize[1]/state.norders
              max = state.plotwinsize[1]-state.scrollsize[1]
              val = (current[1]-offset) > 0
              widget_control, state.plotwin,SET_DRAW_VIEW=[0,val]
              
           end
           
           else:
           
        endcase
        
     endif

  endelse

  cont:
  widget_control, event.top, SET_UVALUE=state, /NO_COPY
  
end
;
;===============================================================================
;
pro xcombspec_resizeevent,event

  widget_control, event.top, GET_UVALUE = state, /NO_COPY
  widget_control, event.id,  GET_UVALUE = uvalue
  
  widget_control, state.xcombspec_base, TLB_GET_SIZE=size
  
  state.plotwinsize[0]=size[0]-state.buffer[0]
  state.scrollsize[0] =state.plotwinsize[0]
  state.scrollsize[1]=size[1]-state.buffer[1]
  
  state.plotwinsize[1] = state.scrollsize[1] > state.pixpp*state.norders

  xcombspec_modwinsize,state
  xcombspec_plotupdate,state
  
  widget_control, state.xcombspec_base, SET_UVALUE=state, /NO_COPY
  
end
;
;===============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;===============================================================================
;
pro xcombspec_cleanup,xcombspec_base

  widget_control, xcombspec_base, GET_UVALUE = state, /NO_COPY
  if n_elements(state) ne 0 then begin

     ptr_free, state.modspeccontinue
     
     ptr_free, state.files
     ptr_free, state.spcmask
     ptr_free, state.pixmask
     ptr_free, state.orders
     ptr_free, state.corspec
     ptr_free, state.scaleorder

     ptr_free, state.ospec
     ptr_free, state.wspec
     ptr_free, state.hdrinfo

     ptr_free, state.awave
     ptr_free, state.atrans
     
  endif 
  state = 0B
  
end
;
;===============================================================================
;
pro xcombspec_combinespec,state,CANCEL=cancel

  cancel = 0

  spc     = *state.wspec
  spcmask = *state.spcmask
  pixmask = *state.pixmask
  
  sspec = (state.combineaps eq 1) ? $
          fltarr(state.npix,4,state.norders):$
          fltarr(state.npix,4,state.norders*state.naps)
  
  if state.combinestat le 2 then begin
     
     thresh = mc_cfld(state.rthresh_fld,4,/EMPTY,CANCEL=cancel)
     if cancel then return
     
  endif

  for i = 0, state.finalnaps-1 do begin
     
     for j = 0, state.norders-1 do begin
        
        z = where(spcmask[*,j,i] eq 1,count)
        spec = reform(spc[i].(j)[*,1,z])
        err  = reform(spc[i].(j)[*,2,z])
        flag = reform(spc[i].(j)[*,3,z])
        mask = reform(pixmask[i].(j))
                
        case state.combinestat of
           
           0: begin

              mc_meancomb,spec,mean,mvar,DATAVAR=err^2,MASK=mask, $
                          ROBUST=thresh,OGOODBAD=ogoodbad,/SILENT,CANCEL=cancel
              if cancel then return

           end
           
           1: begin
              
              mc_meancomb,spec,mean,mvar,/RMS,MASK=mask,ROBUST=thresh,$
                          OGOODBAD=ogoodbad,/SILENT,CANCEL=cancel
              if cancel then return

           end
           
           2: begin
              
              mc_meancomb,spec,mean,mvar,MASK=mask,ROBUST=thresh,$
                          OGOODBAD=ogoodbad,/SILENT,CANCEL=cancel
              if cancel then return

           end
           
           3: begin
              
              mc_meancomb,spec,mean,mvar,DATAVAR=err^2,MASK=mask,/SILENT, $
                          CANCEL=cancel
              if cancel then return
              
           end
           
           4: begin
              
              mc_meancomb,spec,mean,mvar,/RMS,MASK=mask,/SILENT,CANCEL=cancel
              if cancel then return
              
           end
            
           5: begin
              
              mc_meancomb,spec,mean,mvar,MASK=mask,/SILENT,CANCEL=cancel
              if cancel then return
              
           end
           
           6: begin
              
              mc_medcomb,spec,mean,mvar,/MAD,MASK=mask,CANCEL=cancel
              if cancel then return
              
           end
           
           7: begin
              
              mc_medcomb,spec,mean,mvar,MASK=mask,CANCEL=cancel
              if cancel then return
              
           end
           
           8: begin
                            
              mean = total(spec*mask,2,/NAN)
              mvar = total(mask*double(err)^2,2,/NAN)
              
           end
           
        endcase
        
        sspec[*,0,j*state.finalnaps+i] = reform(spc[i].(j)[*,0,0])
        sspec[*,1,j*state.finalnaps+i] = mean
        sspec[*,2,j*state.finalnaps+i] = sqrt(mvar)

;  Combine the flags

        tmp = mask*byte(flag)
        if state.combinestat le 2 then tmp = tmp*ogoodbad
        result = mc_combflagstack(tmp[*,z],CANCEL=cancel)
        if cancel then return
        sspec[*,3,j*state.finalnaps+i] = result
        
     endfor

  endfor
  
  *state.combspec = sspec
  
end
;
;===============================================================================
;
pro xcombspec_correctspec,state

  cancel = 0
  
  if (*state.modspeccontinue)[state.ap] ge 3 then begin
     
     ok = dialog_message([['Cannot perform this operation again.'],$
                          ['Please reload spectra and start over.']],/ERROR,$
                         DIALOG_PARENT=state.xcombspec_base)
     cancel = 1
     return
     
  endif
  
  spc  = (*state.wspec)[state.ap]
  
  for i = 0, state.norders-1 do begin
     
     spec = reform(spc.(i)[*,1,*])
     err  = reform(spc.(i)[*,2,*])
     pixmask = reform((*state.pixmask).(i))

     nstack = mc_speccor(spec,4,IERRSTACK=err,OERRSTACK=nerrstack, $
                         SPECMASK=(*state.spcmask)[*,i,state.ap], $
                         PIXMASK=(*state.pixmask).(i), $
                         CORRECTIONS=corrections,CANCEL=cancel)
     if cancel then return

     spc.(i)[*,1,*] = nstack
     spc.(i)[*,2,*] = nerrstack
     
  endfor
  

  (*state.wspec)[state.ap] = spc
  (*state.corspec)[state.ap] = 1
  xcombspec_combinespec,state,CANCEL=cancel
  if cancel then return
  xcombspec_plotupdate,state
  
  (*state.modspeccontinue)[state.ap] = 3
  
end
;
;===============================================================================
;
pro xcombspec_writefile,state

  path = mc_cfld(state.path_fld,7,CANCEL=cancel)
  if cancel then return

  outname = mc_cfld(state.outfile_fld,7,CANCEL=cancel,/EMPTY)
  if cancel then return

;  Make new hdr.

  if keyword_set(state.basic) then begin

     avehdr = (*state.hdrinfo)[0]
     history = ''
     napcomb = 0

  endif else begin
     
     case state.combtype of

        0: begin                ;  Standard spextool
           
           avehdr = mc_avehdrs(*state.hdrinfo,CANCEL=cancel)
           if cancel then return
           
;  Update some keywords
           
           avehdr.vals.creprog = 'xcombspec'
           avehdr.vals.filename = 'outname'+'.fits'
           history = ''
           napcomb = 0
           
        end
        
        1: begin                ; two-aperture combine
           
           avehdr = (*state.hdrinfo)[0]
           avehdr.vals.totitime = 2*avehdr.vals.totitime
           avehdr.vals.creprog = 'xcombspec'
           avehdr.vals.filename = 'outname'+'.fits'
           history = ''        
           napcomb = 2
           
        end
        
        2: begin                ; xtellcor
           
;  Create a fake avehdr that we can use
           
;  First grab useful things from the first header
           
           keywords = ['INSTR','MODE','RA','DEC','WAVETYPE','NAPS','NORDERS', $
                       'ORDERS','SLTH_PIX','SLTH_ARC','SLTW_PIX','SLTW_ARC', $
                       'RP','XUNITS','YUNITS','XTITLE','YTITLE']
           
           hdrinfo = mc_gethdrinfo((*state.hdrinfo).(0),keywords,CANCEL=cancel)
           if cancel then return
           
           vals = hdrinfo.vals
           coms = hdrinfo.coms
           
;  Now merge a few useful things
           
           for i = 0,state.nfiles-1 do begin
              
              tmp = fxpar((*state.hdrinfo).(i),'AVE_DATE')
              dates = (i eq 0) ? tmp:dates+','+tmp
              
           endfor
           
           for i = 0,state.nfiles-1 do begin
              
              tmp = fxpar((*state.hdrinfo).(i),'TOTITIME')
              itots = (i eq 0) ? tmp:(itots+tmp)
              
           endfor
           
           for i = 0,state.nfiles-1 do begin
              
              tmp = strtrim(fxpar((*state.hdrinfo).(i),'AVE_AM'),2)
              airmasses = (i eq 0) ? tmp:airmasses+','+tmp
              
           endfor                
           
; Take them on the vals and coms structures and recombine
           
           vals = create_struct(vals,'DATES',dates,'TOTITIME',itots, $
                                'AVE_AMS',airmasses)
           coms = create_struct(coms,'DATES',' Dates of observations in UTC',$
                                'TOTITIME',' Total exposure time (sec)',$
                                'AVE_AMS',' Average airmasses')
           avehdr = {vals:vals,coms:coms}
           history = ''
           
           napcomb = 0
           
        end
        
     endcase

  endelse

  avehdr.vals.NAPS = state.finalnaps

;  Now get writing
    
  fxhmake,newhdr,*state.combspec
  ntags = n_tags(avehdr.vals)
  names = tag_names(avehdr.vals)

;  The extra loop is because the history has multiple values
  
  for i = 0, ntags - 1 do begin
     
     if names[i] eq 'HISTORY' then begin

        sxaddhist,(avehdr.vals.(i)),newhdr
        continue

     endif
     
     for k = 0,n_elements(avehdr.vals.(i))-1 do begin
        
        if size((avehdr.vals.(i))[k],/TYPE) eq 7 and $
           strlen((avehdr.vals.(i))[k]) gt 68 then begin
           
           fxaddpar,newhdr,names[i],(avehdr.vals.(i))[k],(avehdr.coms.(i))[k]
           
        endif else sxaddpar,newhdr,names[i],(avehdr.vals.(i))[k], $
                            (avehdr.coms.(i))[k]
        
     endfor
     
  endfor

;  Write new keywords

  if state.userkeywords[0] ne '' then before = state.userkeywords[0]

  sxaddpar,newhdr,'NSPCOMB',state.nfiles,' Number of spectra files combined', $
           BEFORE=before
  sxaddpar,newhdr,'NAPCOMB',napcomb,' Number of apertures combined',$
           BEFORE=before  

;  Create xcombspec history
      
  for i = 0, state.finalnaps-1 do begin

;  Basic combination history
     
     history=history+'The spectra in '+strjoin((*state.files),', ')+$
             ' were combined using a '+state.combstats[state.combinestat]

     if state.combinestat le 2 then begin
        
        thresh = mc_cfld(state.rthresh_fld,4,/EMPTY,CANCEL=cancel)
        if cancel then return
        history = history+' with a threshold of '+ $
                  string(thresh,FORMAT='(G0.5)')+'.  '
        
     endif else history = history+'.'

;  Aperture specific history
     
     history = history+'  Aperture '+string(i+1,FORMAT='(i2.2)')+$
     ' Modifications: '

     add = ''

     if finite((*state.scaleorder)[i]) eq 1 then begin
        
        add = 'The scale factors were determined using order '+$
              string((*state.scaleorder)[i],FORMAT='(I3.3)')+' and are '+ $
              strjoin(strtrim((*state.scales)[*,i],2),', ')+'.  '
        
     endif 
     
     for j = 0,state.norders-1 do begin
        
        z = where((*state.spcmask)[*,j,i] eq 0,count)
        if count ne 0 then begin
           
           add = add+'The spectrum(a) from file(s) '+ $
                 strjoin((*state.files)[z],', ')+' was (ere) ' + $
                 'removed from order '+ $
                 string(total((*state.orders)[j]),FORMAT='(i3)')+'.  '
           
        endif
        
     endfor

     if state.instrument eq 'NIHTS' then begin
        
        add = 'The spectral pixel shifts are '+ $
              strjoin(string((*state.offsets)[*,i],FORMAT='(f+5.2)'),', ')+'.  '



     endif

     
     
     if (*state.corspec)[i] then $
        add = add+'The spectral shapes have been corrected.  '

     history = (add eq '') ? history+'None.':history+add
     
  endfor

  sxaddhist,' ',newhdr
  sxaddhist,'######################## Xcombspec History ' + $
            '########################',newhdr
  sxaddhist,' ',newhdr
  
  history = mc_splittext(history,67,CANCEL=cancel)
  if cancel then return
  sxaddhist,history,newhdr

  writefits,path+outname+'.fits',*state.combspec,newhdr
  
  xvspec,path+outname+'.fits',/PLOTLINMAX,GROUP_LEADER=state.xcombspec_base
  
end
;
;===============================================================================
;
pro xcombspec_loadspec,state

;  Get path and file names
    
  path = mc_cfld(state.path_fld,7,CANCEL=cancel)
  if cancel then return
  
  files = mc_cfld(state.cfiles_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
;  Construct array of full file names
  
  index    = (state.filereadmode eq 'Index') ? 1:0
  filename = (state.filereadmode eq 'Filename') ? 1:0
  if index then prefix = mc_cfld(state.inprefix_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  files = mc_fsextract(files,INDEX=index,FILENAME=filename,CANCEL=cancel)
  if cancel then return
  
  fullpaths = mc_mkfullpath(path,files,INDEX=index,FILENAME=filename,$
                            NI=state.nint,PREFIX=prefix,SUFFIX='.fits',$
                            WIDGET_ID=state.xcombspec_base,/EXIST,$
                            CANCEL=cancel)
  if cancel then return

;  Read the first spectrum to get initial data

  mc_readspec,fullpaths[0],first,hdr,obsmode,start,stop,norders,naps,orders,$
              xunits,yunits,slith_pix,slith_arc,slitw_pix,slitw_arc,$
              rp,airmass,xtitle,ytitle,instr,xranges,program,/SILENT, $
              CANCEL=cancel

  program = strtrim(program,2)
  
  state.npix     = n_elements(first[*,0,0])
  state.norders  = norders
  state.naps     = naps
  state.xtitle   = xtitle
  lidx           = strpos(ytitle,'(')
  ridx           = strpos(ytitle,')')
  yunits         = strmid(ytitle,lidx+1,ridx-lidx-1)
  state.ytitle   = [ytitle,'!5Uncertainty ('+yunits+')','!5S/N']
  *state.orders  = orders
  state.nfiles   = n_elements(fullpaths)

  *state.modspeccontinue = intarr(naps)
  
;  Figure out combine type

;  Get header stuff set up

  if ~keyword_set(state.basic) then begin
  
     case program of
        
        'xspextool': begin
           
           state.combineaps = (n_elements(fullpaths) eq 1) ? 1:0        
           state.combtype = (state.combineaps eq 1) ? 1:0
           
           hdrinfo = mc_gethdrinfo(hdr,state.keywords,/IGNOREMISSING, $
                                   CANCEL=cancel)
           if cancel then return
           hdrinfo = replicate(hdrinfo,state.nfiles)
                       
        end
        
        'xtellcor': begin
           
           state.combineaps = 0
           state.combtype = 2
           hdrinfo = {hdr1:hdr}
           
        end
        
        else: begin
           
           ok = dialog_message('Cannot combine this kinda of data.',/ERROR,$
                               DIALOG_PARENT=state.xcombspec_base)
           cancel = 1
           return
           
        end
        
     endcase

  endif else begin

     state.combineaps = (n_elements(fullpaths) eq 1) ? 1:0        
     state.combtype = (state.combineaps eq 1) ? 1:0
     
     hdrinfo = mc_gethdrinfo(hdr,state.keywords,/IGNOREMISSING,CANCEL=cancel)
     if cancel then return
     hdrinfo = replicate(hdrinfo,state.nfiles)
     

  endelse

     
;  Update order buttons

  widget_control, state.scaleorder_dl,$
                  SET_VALUE=reverse(string(*state.orders,FORMAT='(i3)'))

  widget_control, state.pruneorder_dl,$
                  SET_VALUE=reverse(string(*state.orders,FORMAT='(i3)'))

  widget_control, state.order_dl,$
                  SET_VALUE=reverse(string(*state.orders,FORMAT='(i3)'))

  widget_control, state.scale_base, SENSITIVE=1
  widget_control, state.prune_base, SENSITIVE=1
  
;  Modify plot window according to the number of orders
  
  state.plotwinsize[1] = state.plotwinsize[1] > state.pixpp*state.norders
  xcombspec_modwinsize,state

;  Get the atmospheric transmission if the data are from NIHTS

;  Load the atmospheric transmission if need be

  if state.instrument eq 'NIHTS' and rp ne 0 and $
     (xunits eq 'um' or xunits eq 'nm' or xunits eq 'A') then begin

;  Get the resolutions available
     
     files = file_basename(file_search(filepath('atran*.fits', $
                                                ROOT_DIR=state.spextoolpath, $
                                                SUBDIR='data')))
     
     nfiles = n_elements(files)
     rps = lonarr(nfiles)
     for i =0,nfiles-1 do rps[i] = long(strmid( $
        file_basename(files[i],'.fits'),5))
     
     min = min(abs(rps-rp),idx)
     
     spec = readfits(filepath('atran'+strtrim(rps[idx],2)+'.fits', $
                              ROOT_DIR=state.spextoolpath, $
                              SUBDIR='data'),/SILENT)
     
     *state.awave = reform(spec[*,0])
     *state.atrans = reform(spec[*,1])

  endif else begin

     *state.awave = 0
     *state.atrans = 0
     
  endelse
       
;  We are now going to be build two arrays of structures to store the
;  data, spc, and the pixel mask, pixmask.  Each element of each array
;  will contain a structure where each tag contains the spectrum or
;  pixel mask for an order.

;  Set up arrays checking to see if combining apertures

  if state.combineaps then begin

     state.finalnaps = 1
     state.nspec   = 2*state.nfiles

     *state.scales = make_array(state.nspec,/FLOAT,VALUE=1)
     *state.offsets = make_array(state.nspec,/FLOAT,VALUE=1)     
     array   = fltarr(state.npix,4,state.nspec)+!values.f_nan
     spcmask = make_array(state.nspec,state.norders,/INT,VALUE=1)
     pmask   = make_array(state.npix,state.nspec,/INT,VALUE=1)

     widget_control, state.aperture_dl, SET_VALUE='1'     
     widget_control, state.aperture_dl, SENSITIVE=0

  endif else begin
     
     state.finalnaps = state.naps
     state.nspec     = state.nfiles
     
     *state.scales = make_array(state.nfiles,state.finalnaps,/FLOAT,VALUE=1)
     *state.offsets = make_array(state.nfiles,state.finalnaps,/FLOAT,VALUE=1) 
     array   = fltarr(state.npix,4,state.nfiles)*!values.f_nan
     spcmask = intarr(state.nfiles,state.norders,state.finalnaps)+1
     pmask   = intarr(state.npix,state.nfiles)+1

     widget_control, state.aperture_dl, $
                     SET_VALUE=strcompress(indgen(state.naps)+1, /RE)     
     widget_control, state.aperture_dl, SENSITIVE=(state.naps eq 1) ? 0:1

  endelse

  *state.scaleorder   = make_array(state.naps,/FLOAT,VALUE=!values.f_nan)
  state.scaleorderidx = 0
  state.pruneorderidx = 0
  *state.corspec      = intarr(state.finalnaps)
  
;  Create data structure for a single aperture
  
  key     = 'Order'+string(00,FORMAT='(i3.3)')
  spc     = create_struct(key,array)
  pixmask = create_struct(key,pmask)
  
  for i = 1, state.norders-1 do begin
     
     key     = 'Order'+string(i,FORMAT='(i3.3)')
     spc     = create_struct(spc,key,array)
     pixmask = create_struct(pixmask,key,pmask)
     
  endfor

;  If there are more than 1 apertures replicate the data structure.
  
  spc     = replicate(spc,state.finalnaps)
  pixmask = replicate(pixmask,state.finalnaps)
  
;  Load the data

  for i = 0, state.nfiles-1 do begin
     
     data = readfits(fullpaths[i],hdr,/SILENT)

;     if ~keyword_set(state.basic) then begin
     
        if state.combtype eq 2 then begin ;  xtellcor 
           
           hdrinfo = (i eq 0) ? {hdr1:hdr}: $
                     create_struct(hdrinfo,'hdr'+strtrim(i+1,2),hdr)
           
        endif else begin
           
           hdrinfo[i] = mc_gethdrinfo(hdr,state.keywords,/IGNOREMISSING, $
                                      CANCEL=cancel)
           if cancel then return
           
        endelse

;     endif
        
;  Add a flag array if need be
     
     data = mc_caaspecflags(data,CANCEL=cancel)
     if cancel then return

;  Now start the loop over the orders
     
     for j = 0, state.norders-1 do begin

        if ~state.combineaps then begin
           
           for k = 0, state.naps-1 do begin
              
              z = where(finite(first[*,0,j*state.naps+k]) eq 1)

              mc_interpspec,data[z,0,j*state.naps+k],data[z,1,j*state.naps+k], $
                            first[z,0,j*state.naps+k],newflux,newerror, $
                            IYERROR=data[z,2,j*state.naps+k],LEAVENANS=1,$
                            CANCEL=cancel
              if cancel then return

              mc_interpflagspec,data[z,0,j*state.naps+k], $
                                byte(data[z,3,j*state.naps+k]),$
                                first[z,0,j*state.naps+k], $
                                newflag,CANCEL=cancel
              if cancel then return
              
              spc[k].(j)[z,0,i] = first[z,0,j*state.naps+k]
              spc[k].(j)[z,1,i] = newflux
              spc[k].(j)[z,2,i] = newerror
              spc[k].(j)[z,3,i] = float(newflag)
              
           endfor
           
        endif else begin

           for k = 0,state.naps-1 do begin
              
              z = where(finite(first[*,0,j*state.naps+k]) eq 1)

              mc_interpspec,data[z,0,j*state.naps+k],data[z,1,j*state.naps+k], $
                            first[z,0,j*state.naps+k],newflux,newerror, $
                            IYERROR=data[z,2,j*state.naps+k],LEAVENANS=1,$
                            CANCEL=cancel
              if cancel then return

              mc_interpflagspec,data[z,0,j*state.naps+k],$
                                byte(data[z,3,j*state.naps+k]),$
                                first[z,0,j*state.naps+k], $
                                newflag,CANCEL=cancel
              if cancel then return
             
                spc[0].(j)[z,0,i*state.naps+k] = first[z,0,j*state.naps+k]
                spc[0].(j)[z,1,i*state.naps+k] = newflux
                spc[0].(j)[z,2,i*state.naps+k] = newerror
                spc[0].(j)[z,3,i*state.naps+k] = float(newflag)

             endfor
           
        endelse
        
     endfor
          
  endfor

  *state.ospec    = spc
  *state.wspec    = spc
  *state.spcmask  = spcmask
  *state.pixmask  = pixmask
  if ~keyword_set(state.basic) then *state.hdrinfo  = hdrinfo

;  state.itot = itot
  state.ap = 0
  
;  Create filename string for output FITS header later
  
  for i = 0,state.nfiles-1 do begin
     
     file = strmid(fullpaths[i],strpos(fullpaths[i],'/',/REVERSE_S)+1)
     sfile = (i eq 0) ? file:[sfile,file]
     
  endfor
  *state.files = sfile
  
;  Unfreeze the widget
  
  state.freeze = 0
  xcombspec_combinespec,state,CANCEL=cancel
  if cancel then return
  xcombspec_plotupdate,state

end
;
;===============================================================================
;
pro xcombspec_prunespec,state

  cancel = 0

  index = state.pruneorderidx

  spc    = (*state.ospec)[state.ap]
  zorder = total(where(*state.orders eq (*state.orders)[index]))
  
  x = reform(spc.(zorder)[*,0,0])
  y = reform(spc.(zorder)[*,1,*])

  s = size(y,/DIMEN)

  scales = rebin(reform((*state.scales)[*,state.ap],1,s[1]),s[0],s[1])

  xmc_maskspec,reform(spc.(zorder)[*,0,0]), $
               reform(spc.(zorder)[*,1,*])*scales,$
               specmask,pixmask,XTITLE=state.xtitle, $
               YTITLE=state.ytitle[0], $
               GROUP_LEADER=state.xcombspec_base,CANCEL=cancel
 if cancel then return

  (*state.spcmask)[*,index,state.ap] = specmask
  (*state.pixmask)[state.ap].(zorder) = pixmask

  
  widget_control, /HOURGLASS
  
  xcombspec_combinespec,state,CANCEL=cancel
  if cancel then return
  xcombspec_plotupdate,state

end
;
;===============================================================================
;
pro xcombspec_modwinsize,state


  widget_control, state.col2_base, UPDATE=0
  widget_control, state.plotwin, /DESTROY
  
  if state.plotwinsize[1] le state.scrollsize[1] then begin

     state.plotwin = widget_draw(state.col2_base,$
                                   XSIZE=state.plotwinsize[0],$
                                   YSIZE=state.plotwinsize[1],$
                                   UVALUE='Plot Window',$
                                   /TRACKING_EVENTS,$
                                   /KEYBOARD_EVENTS,$
                                   /BUTTON_EVENTS,$
                                   EVENT_PRO='xcombspec_plotwinevent')

  endif else begin

     state.plotwin = widget_draw(state.col2_base,$
                                   XSIZE=state.plotwinsize[0],$
                                   YSIZE=state.plotwinsize[1],$
                                   X_SCROLL_SIZE=state.scrollsize[0],$
                                   Y_SCROLL_SIZE=state.scrollsize[1],$
                                   /SCROLL,$
                                   UVALUE='Plot Window',$
                                   /TRACKING_EVENTS,$
                                   /KEYBOARD_EVENTS,$
                                   /BUTTON_EVENTS,$
                                   EVENT_PRO='xcombspec_plotwinevent')


  endelse

  widget_control, state.plotwin, GET_VALUE = wid
  state.plotwin_wid = wid
  wset, wid
  erase, COLOR=20

  widget_control, state.col2_base, UPDATE=1
  
  widget_geom = widget_info(state.xcombspec_base, /GEOMETRY)
  
  state.buffer[0]=widget_geom.xsize-state.scrollsize[0]
  state.buffer[1]=widget_geom.ysize-state.scrollsize[1]
  

  
  wdelete, state.pixmap_wid
  window, /FREE, /PIXMAP,XSIZE=state.plotwinsize[0],$
          YSIZE=state.plotwinsize[1]
  state.pixmap_wid = !d.window
  

end
;
;===============================================================================
;
pro xcombspec_plotspec,state

  spc  = *state.wspec
  mask = *state.spcmask
  pixmask = *state.pixmask
  
  !p.multi[0] = state.norders
  !p.multi[2] = state.norders
  
  case state.spectype of
     
     0: ytitle = state.ytitle[0]
     
     1: ytitle = state.ytitle[1]
     
     2: ytitle = state.ytitle[2]
     
  endcase

  charsize = state.charsize
  if state.norders ge 3 then charsize = charsize*2.0
  
  if state.plottype eq 0 then begin
     
     for i = 0, state.norders-1 do begin
        
        j     = state.norders-1-i

        title= '!5Order '+strtrim(string((*state.orders)[j],FORMAT='(i3)'),2)
        aps = (state.combineaps eq 1) ? ', Apertures 1&2':', Aperture '+ $
              strtrim(string(state.ap+1,FORMAT='(i2)'),2)
        title = title+aps

;  Get plot range.
        
        case state.spectype of 
           
           0: spec = reform(spc[state.ap].(j)[*,1,*])
           
           1: spec = reform(spc[state.ap].(j)[*,2,*])
           
           2: spec = reform(spc[state.ap].(j)[*,1,*] / $
                            spc[state.ap].(j)[*,2,*])
           
        end
        
        mc_medcomb,spec,med
        yrange = [0.4*min(med,/NAN,MAX=max),1.5*max]
        
;  Plot spectra    
        
        plot,spc[state.ap].(j)[*,0,0],spec[*,0],$
             /XSTY,/YSTY,YRANGE=yrange,/NODATA,TITLE=title,$
             XTITLE=state.xtitle,YTITLE=ytitle, $
             CHARSIZE=charsize
        
        for k = 0, state.nspec-1 do begin
            
            if mask[k,j,state.ap] eq 1 then begin

                flux = spec[*,k]
                z = where(pixmask[state.ap].(j)[*,k] eq 0,cnt)
                if cnt ne 0 then flux[z] = !values.f_nan

                oplot,spc[state.ap].(j)[*,0,k],flux,COLOR=state.colors[k], $
                      LINESTYLE=state.linestyles[k],PSYM=10

          endif
            
        endfor
        
    endfor
     
  endif

  if state.plottype eq 1 then begin
     
     wave = reform((*state.combspec)[*,0,*])
     
     case state.spectype of 
        
        0: spec = reform((*state.combspec)[*,1,*])
        
        1: spec = reform((*state.combspec)[*,2,*])
        
        2: spec = reform((*state.combspec)[*,1,*] / $
                         (*state.combspec)[*,2,*])
        
     end
     
     for i = 0, state.norders-1 do begin
        
        j     = state.norders-1-i
        title = 'Order '+string((*state.orders)[j],FORMAT='(i3)')
        
;  Get plot range.
        
        yrange = [0.4*min(spec[*,j*state.finalnaps+state.ap],/NAN,MAX=max), $
                  1.5*max]
        
;  Plot spectra    
        
        plot,wave[*,j*state.finalnaps+state.ap], $
             spec[*,j*state.finalnaps+state.ap],$
             /XSTY,/YSTY,YRANGE=yrange,CHARSIZE=charsize,$
             TITLE=title, XTITLE=state.xtitle,YTITLE=ytitle,PSYM=10
        
     endfor
     
  endif
  
  !p.multi=0
  delvarx,spc

end
;
;===============================================================================
;
pro xcombspec_plotupdate,state

  wset,state.pixmap_wid
  erase
  polyfill,[0,0,1,1,0],[0,1,1,0,0],COLOR=20,/NORM
  xcombspec_plotspec,state
  
  wset, state.plotwin_wid
  device, COPY=[0,0,state.plotwinsize[0],state.plotwinsize[1],0,0,$
                state.pixmap_wid]

end
;
;===============================================================================
;
pro xcombspec_shiftspec,state

  if (*state.modspeccontinue)[state.ap] gt 1 then begin
     
     ok = dialog_message([['Cannot perform this operation again.'],$
                          ['Please reload spectra and start over.']],/ERROR,$
                         DIALOG_PARENT=state.xcombspec_base)
     cancel = 1
     return
     
  endif

  
  spc = (*state.wspec)[state.ap]
  
  if state.combineaps then begin
     
     files = *state.files+' Ap '+ $
             string(findgen(state.naps)+1,FORMAT='(I2.2)')

     nspec = n_elements(files)*2
     
  endif else begin

     files = *state.files
     nspec = n_elements(files)
     
  endelse

;  Interpolate the atmosphere

  linterp,*state.awave,*state.atrans,reform(spc.(0)[*,0,0]),newtrans, $
          MISSING=!values.f_nan

  
  xmc_getoffset,reform(spc.(0)[*,0,0]),reform(spc.(0)[*,1,*]),files, $
                reform(spc.(0)[*,0,0]),newtrans,sstack,offsets,wrange,$
                GROUP_LEADER=state.xcombspec_base,XTITLE=state.xtitle, $
                YTITLE=state.ytitle[0],CANCEL=cancel
  if cancel then return

  widget_control, /HOURGLASS
  ndat = n_elements(spc.(0)[*,0,0])
  pix = findgen(ndat)

  s = reform(spc.(0)[*,1,*])
  u = reform(spc.(0)[*,2,*])
     
  for k = 0, nspec-1 do begin
     
     linterp,pix,s[*,k],pix-total(offsets[k]),nd
     linterp,pix,u[*,k],pix-total(offsets[k]),nu
     
     s[*,k] = nd
     u[*,k] = nu
     
  endfor
  
  spc.(0)[*,1,*] = s
  spc.(0)[*,2,*] = u

;  Store scales
     
  (*state.offsets)[*,state.ap] = offsets
  
  (*state.wspec)[state.ap] = spc
  
  xcombspec_combinespec,state,CANCEL=cancel
  if cancel then return
  xcombspec_plotupdate,state

  (*state.modspeccontinue)[state.ap]=2
  
  
end
;
;===============================================================================
;
Pro xcombspec_scalespec,state

  cancel = 0

  if (*state.modspeccontinue)[state.ap] gt 0 then begin
     
     ok = dialog_message([['Cannot perform this operation again.'],$
                          ['Please reload spectra and start over.']],/ERROR,$
                         DIALOG_PARENT=state.xcombspec_base)
     cancel = 1
     return
     
  endif

  index = state.scaleorderidx

  spc = (*state.ospec)[state.ap]
  good = where((*state.spcmask)[*,index,state.ap] eq 1)
  zorder = total(where(*state.orders eq (*state.orders)[index]))
  (*state.scaleorder)[state.ap] = (*state.orders)[index]

  if state.combineaps then begin
     
     files = *state.files+' Ap '+ $
             string(findgen(state.naps)+1,FORMAT='(I2.2)')
     
  endif else files = *state.files


  xmc_scalespec,reform(spc.(zorder)[*,0,0]),reform(spc.(zorder)[*,1,*]), $
                files,reform((*state.spcmask)[*,index,state.ap]), $
                scales,wrange,GROUP_LEADER=state.xcombspec_base, $
                XTITLE=state.xtitle,YTITLE=state.ytitle[0],CANCEL=cancel
  if cancel then return
  
  widget_control, /HOURGLASS
  for i = 0, state.norders-1 do begin
     
     s = reform(spc.(i)[*,1,*])
     e = reform(spc.(i)[*,2,*])
     
     for k = 0, n_elements((*state.spcmask)[*,i,state.ap])-1 do begin
        
        s[*,k] = scales[k] * s[*,k]
        e[*,k] = scales[k] * e[*,k]
        
     endfor
     
     spc.(i)[*,1,*] = s
     spc.(i)[*,2,*] = e        
     
;  Store scales
     
     (*state.scales)[*,state.ap] = scales
     
  endfor

  (*state.wspec)[state.ap] = spc
  
  
  xcombspec_combinespec,state,CANCEL=cancel
  if cancel then return
  xcombspec_plotupdate,state
  
  (*state.modspeccontinue)[state.ap]=1
  
end
;
;===============================================================================
;
pro xcombspec,instrument,BASIC=basic,CANCEL=cancel

  mc_mkct

  cleanplot,/SILENT
  device, RETAIN=2

;  Get spextool and instrument information 
  
  mc_getspextoolinfo,spextoolpath,packagepath,spextool_keywords,instrinfo, $
                     notirtf,version,CANCEL=cancel
  if cancel then return
  
;  Set the fonts

  mc_getfonts,buttonfont,textfont,CANCEL=cancel
  if cancel then return

;  Build three structures that will hold important info.
  
  state = {ap:0,$
           aperture_dl:0L,$
           atrans:ptr_new(2),$
           awave:ptr_new(2),$
           basic:keyword_set(BASIC),$
           box2_base:0L,$
           box4_base:0L,$
           buffer:[0,0],$
           cfiles_fld:[0L,0L],$
           charsize:1.5,$
           col2_base:0L,$
           colors:reform(rebin(findgen(15)+1,15,6),15*6),$           
           combspec:ptr_new(2),$
           combtype:0,$
           combineaps:0,$
           combinestat:0,$
           combstats:['Robust Weighted Mean','Robust Mean (RMS)', $
                      'Robust Mean (Std Error)','Weighted Mean',$
                      'Mean (RMS)','Mean (Std Error)','Median (MAD)', $
                      'Median (Median Error)','Sum'],$
           correct_base:0L,$
           corspec:ptr_new(2),$
           file_base:0L,$
           filename_fld:[0L,0L],$
           filereadmode:'Index',$
           files:ptr_new(fltarr(2)),$
           finalnaps:0,$
           freeze:1,$
           hdrinfo:ptr_new(strarr(2)),$
           inprefix_fld:[0L,0L],$
           instrument:instrinfo.instrument,$
           itot:0.0,$
           keywords:[spextool_keywords,instrinfo.xcombspec_keywords,'HISTORY'],$
           linestyles:reform(rebin(reform(indgen(6),1,6),15,6),15*6),$
           modspeccontinue:ptr_new(2),$
           modtype_dl:0L,$
           naps:0,$
           nfiles:0,$
           nint:instrinfo.nint,$
           norders:0,$
           npix:0,$
           nspec:0,$
           offsets:ptr_new(2),$           
           order_dl:0L,$
           orders:ptr_new(2),$     
           ospec:ptr_new(fltarr(2)),$
           outfile_fld:[0L,0L],$
           packagepath:packagepath,$
           path_fld:[0L,0L],$
           pixmap_wid:0L,$
           pixmask:ptr_new(2),$
           pixpp:250.0,$
           plottype:0,$
           plotwin:0L,$
           plotwin_wid:0,$
           plotwinsize:get_screen_size()*[0.4,0.65],$
           prune_base:0L,$
           pruneorder_dl:0L,$
           pruneorderidx:0L,$
           rthresh_fld:[0L,0L],$
           scale_base:0L,$
           scales:ptr_new(2),$
           scaleorder_dl:0L,$
           scaleorder:ptr_new(0),$
           scaleorderidx:0L,$
           scrollsize:get_screen_size()*[0.4,0.65],$
           spectype:0,$
           spcmask:ptr_new(2),$
           spextoolpath:spextoolpath,$
           userkeywords:instrinfo.xcombspec_keywords,$
           wspec:ptr_new(fltarr(2)),$
           xcombspec_base:0L,$
           xtitle:'',$
           ytitle:['','','']}

;  Build the widget.

  title = (keyword_set(BASIC) eq 1) ? 'xcombspec '+version:'xcombspec '+ $
          version+' for '+state.instrument
  
  state.xcombspec_base = widget_base(TITLE=title,$
                                       /COLUMN,$
                                       /TLB_SIZE_EVENTS)

     button = widget_button(state.xcombspec_base,$
                            FONT=buttonfont,$
                            VALUE='Quit',$
                            EVENT_PRO='xcombspec_event',$
                            UVALUE='Quit')
     
     row_base = widget_base(state.xcombspec_base,$
                            /ROW)

        col1_base = widget_base(row_base,$
                                EVENT_PRO='xcombspec_event',$
                                /COLUMN)

           box1_base = widget_base(col1_base,$
                                   /COLUMN,$
                                   FRAME=2)

              label = widget_label(box1_base,$
                                   VALUE='1.  Load Spectra',$
                                   /ALIGN_LEFT,$
                                   FONT=buttonfont)
              
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 button = widget_button(row,$
                                        FONT=buttonfont,$
                                        VALUE='Path',$
                                        UVALUE='Path Button')
                 
                 fld = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE=':',$
                                     UVALUE='Path Field',$
                                     XSIZE=25,$
                                     TEXTID=textid)
                 state.path_fld = [fld,textid]               
                 
              bg = cw_bgroup(box1_base,$
                             ['Filename','Index'],$
                             /ROW,$
                             LABEL_LEFT='File Read Mode:',$
                             /RETURN_NAME,$
                             /NO_RELEASE,$
                             UVALUE='Readmode',$
                             FONT=buttonfont,$
                             /EXCLUSIVE,$
                             SET_VALUE=1)
              
              fld = coyote_field2(box1_base,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE='Input Prefix:',$
                                  UVALUE='Input Prefix',$
                                  XSIZE=20,$
                                  VALUE='spectra',$
                                  TEXTID=textid)
              state.inprefix_fld = [fld,textid]
              
              row = widget_base(box1_base,$
                                /ROW,$
                                /BASE_ALIGN_CENTER)
              
                 button = widget_button(row,$
                                        FONT=buttonfont,$
                                        VALUE='Files',$
                                        UVALUE='Spectra Files Button')
                 
                 field = coyote_field2(row,$
                                       LABELFONT=buttonfont,$
                                       FIELDFONT=textfont,$
                                       TITLE=':',$
                                       UVALUE='Spectra Files Field',$
;                                       VALUE='tc1.fits,tc2.fits',$
                                       XSIZE=25,$
                                       /CR_ONLY,$
                                       TEXTID=textid)
                 state.cfiles_fld = [field,textid]
                 
                 load = widget_button(box1_base,$
                                      FONT=buttonfont,$
                                      VALUE='Load Spectra',$
                                      UVALUE='Load Spectra')

         state.box2_base = widget_base(col1_base,$
                                         /COLUMN,$
                                         FRAME=2)

            label = widget_label(state.box2_base,$
                                 VALUE='2.  Modify Spectra',$
                                 /ALIGN_LEFT,$
                                 FONT=buttonfont)

            state.scale_base = widget_base(state.box2_base,$
                                             /ROW,$
                                             /BASE_ALIGN_CENTER)

               state.scaleorder_dl = widget_droplist(state.scale_base,$
                                                       FONT=buttonfont,$
                                                       TITLE='Order:',$
                                                       VALUE='001',$
                                                       UVALUE='Scale Order')

               button = widget_button(state.scale_base,$
                                      FONT=buttonfont,$
                                      VALUE='Scale Spectra',$
                                      UVALUE='Scale Spectra')

               if state.instrument eq 'NIHTS' then begin

                  button = widget_button(state.box2_base,$
                                         FONT=buttonfont,$
                                         VALUE='Shift Spectra',$
                                         UVALUE='Shift Spectra')

               endif
               
               
            state.prune_base = widget_base(state.box2_base,$
                                             /ROW,$
                                             /BASE_ALIGN_CENTER)

               state.pruneorder_dl = widget_droplist(state.prune_base,$
                                                       FONT=buttonfont,$
                                                       TITLE='Order:',$
                                                       VALUE='001',$
                                                       UVALUE='Prune Order')

               button = widget_button(state.prune_base,$
                                      FONT=buttonfont,$
                                      VALUE='Prune Spectra',$
                                      UVALUE='Prune Spectra')

            state.correct_base = widget_base(state.box2_base,$
                                               /ROW,$
                                               /BASE_ALIGN_CENTER)

               button = widget_button(state.box2_base,$
                                      FONT=buttonfont,$
                                      VALUE='Correct Spectral Shape',$
                                      UVALUE='Correct Spectral Shape')
                                           
         
         state.box4_base = widget_base(col1_base,$
                                         /COLUMN,$
                                         FRAME=2)

            label = widget_label(state.box4_base,$
                                 VALUE='3.  Write Spectra',$
                                 /ALIGN_LEFT,$
                                 FONT=buttonfont)
            
            row = widget_base(state.box4_base,$
                              /ROW,$
                              /BASE_ALIGN_CENTER)


               field = coyote_field2(row,$
                                     LABELFONT=buttonfont,$
                                     FIELDFONT=textfont,$
                                     TITLE='Output File:',$
                                     UVALUE='Output File',$
;                                     VALUE='test',$
                                     XSIZE=20,$
                                     /CR_ONLY,$
                                     TEXTID=textid)
               state.outfile_fld = [field,textid]

            combine_button = widget_button(state.box4_base,$
                                           FONT=buttonfont,$
                                           VALUE='Write File',$
                                           UVALUE='Write File')

         state.col2_base = widget_base(row_base,$
                                         EVENT_PRO='xcombspec_event',$
                                         FRAME=2,$
                                         /COLUMN)

            combine_base = widget_base(state.col2_base,$
                                       /ROW,$
                                       /BASE_ALIGN_CENTER)
            
               combine_dl = widget_droplist(combine_base,$
                                         FONT=buttonfont,$
                                         TITLE='Statistic:',$
                                         VALUE=state.combstats,$
                                         UVALUE='Combination Statistic')
               widget_control, combine_dl, $
                               SET_DROPLIST_SELECT=state.combinestat
               
               fld = coyote_field2(combine_base,$
                                   LABELFONT=buttonfont,$
                                   FIELDFONT=textfont,$
                                   TITLE='Robust Threshold:',$
                                   UVALUE='Robust Threshold',$
                                   XSIZE=5,$
                                   VALUE='8.0',$
                                   EVENT_PRO='xcombspec_event',$
                                   /CR_ONLY,$
                                   TEXTID=textid)
               state.rthresh_fld = [fld,textid]
                                 
            plot_base = widget_base(state.col2_base,$
                                    /ROW,$
                                    /BASE_ALIGN_CENTER)
            
               state.order_dl = widget_droplist(plot_base,$
                                                  FONT=buttonfont,$
                                                  TITLE='Order:',$
                                                  VALUE='  1',$
                                                  UVALUE='Plot Order')   
            
               state.aperture_dl = widget_droplist(plot_base,$
                                                     FONT=buttonfont,$
                                                     TITLE='Aperture:',$
                                                     VALUE='1',$
                                                     UVALUE='Aperture')   

               plot_dl = widget_droplist(plot_base,$
                                         FONT=buttonfont,$
                                         TITLE='Plot:',$
                                         VALUE=['Raw','Combined'],$
                                         UVALUE='Plot Type')   

               plot_dl = widget_droplist(plot_base,$
                                         FONT=buttonfont,$
                                         TITLE='Type:',$
                                         VALUE=['Flux','Uncertainty','S/N'],$
                                         UVALUE='Spectra Type')   

            state.plotwin = widget_draw(state.col2_base,$
                                          XSIZE=state.plotwinsize[0],$
                                          YSIZE=state.plotwinsize[1],$
                                          UVALUE='Plot Window',$
                                          EVENT_PRO= $
                                          'xcombspec_plotwinevent')

     button = widget_button(state.xcombspec_base,$
                            FONT=buttonfont,$
                            VALUE='Help',$
                            EVENT_PRO='xcombspec_event',$
                            UVALUE='Help')
     
; Get things running.  Center the widget using the Fanning routine.

  cgcentertlb,state.xcombspec_base

  widget_control, state.xcombspec_base, /REALIZE
  
;  Get plotwin ids
  
  widget_control, state.plotwin, GET_VALUE=wid
  state.plotwin_wid = wid
  wset, wid
  erase, COLOR=20
     
  window, /FREE, /PIXMAP,XSIZE=state.plotwinsize[0],YSIZE=state.plotwinsize[1]
  state.pixmap_wid = !d.window
   
; Start the Event Loop. This will be a non-blocking program.
   
  XManager, 'xcombspec', $
            state.xcombspec_base, $
            /NO_BLOCK,$
            EVENT_HANDLER='xcombspec_resizeevent',$
            CLEANUP='xcombspec_cleanup'
  
;  Get sizes of things now for resizing
  
  widget_control, state.xcombspec_base, TLB_GET_SIZE=result
  
  widget_geom = widget_info(state.xcombspec_base, /GEOMETRY)

  state.buffer[0] = widget_geom.xsize-state.scrollsize[0]
  state.buffer[1] = widget_geom.ysize-state.scrollsize[1]
  
  
; Put state variable into the user value of the top level base.
  
  widget_control, state.xcombspec_base, SET_UVALUE=state, /NO_COPY
  
end






