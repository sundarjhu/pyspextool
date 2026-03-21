;
;=============================================================================
;
; ----------------------------Support procedures------------------------------ 
;
;=============================================================================
;
pro ishellcals2dxd_mkflat,w,flatinfo,DISPLAY=display,CANCEL=cancel

  cancel = 0

  widget_control, /HOURGLASS
  
  common xspextool_state

;  Keeping the SpeX way of doing things in case we eventually go with
;  a table calibration method
  
  if n_params() eq 1 then begin

     flatinfo = strarr(2)

     index    = (state.r.filereadmode eq 'Index') ? 1:0
     filename = (state.r.filereadmode eq 'Filename') ? 1:0
     
     if index then begin

        prefix = mc_cfld(state.w.iprefix_fld,7,/EMPTY,CANCEL=cancel)
        if cancel then return

     endif
        
;  Get user inputs.
     
     files = mc_cfld(w.flats_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then return
     files = mc_fsextract(files,INDEX=index,FILENAME=filename,NFILES=nfiles,$
                          CANCEL=cancel)
     if cancel then return

     if index then begin
        
;  Check for duplicate files

        junk = mc_checkdups(state.r.datapath,files,state.r.nint, $
                            WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
        if cancel then return
        
     endif
          
     files = mc_mkfullpath(state.r.datapath,files,INDEX=index, $
                           FILENAME=filename,NI=state.r.nint,PREFIX=prefix, $
                           SUFFIX='*.fits*',WIDGET_ID=state.w.xspextool_base, $
                           /EXIST,CANCEL=cancel)
     if cancel then return

     flatinfo[0] = strjoin(files,',')
     
     flatoname = mc_cfld(w.flatoname_fld,7,/EMPTY,CANCEL=cancel)
     if cancel then goto, out
     
     flatinfo[1] = flatoname
     
  endif
  
  files = strsplit(flatinfo[0],',',/EXTRACT)
  flatoname = (reform(flatinfo[1]))[0]
  ofile = filepath(flatoname+'.fits',ROOT=state.r.calpath)

;  Get modeinfo

  hdr  = headfits(files[0])
  mode = strcompress(fxpar(hdr,'XDTILT'),/RE)

  flatinfofile = filepath(mode+'_flatinfo.fits',ROOT=state.r.packagepath, $
                          SUB='data')
  flatinfo = mc_readflatinfo(flatinfofile,CANCEL=cancel)
  if cancel then return

;  Compute slitw and resolving power

  slitw_arc = float(fxpar(hdr,'SLIT'))
  slitw_pix = slitw_arc / flatinfo.ps

  resolvingpower = round(flatinfo.rpppix*slitw_pix)
  
;  Load the images

  call_procedure,'mc_readishellfits',files,data,hdrinfo,var, $
                 KEYWORDS=state.r.keywords,$
                 BITINFO={lincormax:state.r.lincormax,lincormaxbit:0},$
                 BITMASK=bitmask,ROTATE=flatinfo.rotation,NIMAGES=nimages, $
                 CANCEL=cancel
  if cancel then return

;  Combine flags

  bitmask = mc_combflagstack(bitmask,CANCEL=cancel)
  if cancel then return
 
;  Scale the flats to a common flux level 
 
  mc_scaleimgs,data,var,CANCEL=cancel
  if cancel then return
 
;  Combine the images together
 
  mc_combimgs,data,7,mean,var,/UPDATE,IVCUBE=var,CANCEL=cancel
  if cancel then return

;  Figure out any offsets

  tmp = mc_cfld(state.w.flatfieldoffset_fld,4,/EMPTY,CANCEL=canel)
  if cancel then return

  if tmp eq 0.0 then begin
  
;  Adjust the guess positions
     
     adj = mc_adjustguesspos(flatinfo.edgecoeffs,flatinfo.xranges,mean, $
                             rotate(flatinfo.omask,flatinfo.rotation), $
                             flatinfo.orders,flatinfo.ycororder,3,CANCEL=cancel)
     if cancel then return

  endif else begin

     xspextool_message,'Adjusting guess positions by '+strtrim(tmp,2)+' pixels.'
     
     adj = mc_adjustguesspos(flatinfo.edgecoeffs,flatinfo.xranges,mean, $
                             rotate(flatinfo.omask,flatinfo.rotation), $
                             flatinfo.orders,flatinfo.ycororder,3,tmp, $
                             CANCEL=cancel)
     if cancel then return
     
  endelse
     
;  Locate orders
  
  if ~keyword_set(NODISPLAY) then begin
     
     ximgtool, mean, SCALE='Hist Eq',WID=wid,GROUP_LEADER=group_leader, $
               BUFFER=1,STDIMAGE=state.r.stdimage, $
               BITMASK={bitmask:bitmask,plot1:[0,2]}, $
               PLOTWINSIZE=state.r.plotwinsize,PANNER=0,MAG=0, $
               /ZTOFIT,POSITION=[1,0]

  endif

;  Locate the orders

  mc_findorders,mean,adj.guesspos,adj.xranges,flatinfo.step, $
                flatinfo.slith_range,flatinfo.edgedeg,flatinfo.ybuffer, $
                flatinfo.flatfrac,flatinfo.comwin,edgecoeffs,xranges, $
                WID=wid,CANCEL=cancel
  if cancel then return
  
;  Normalize the flat

  if ~keyword_set(NONORM) then begin
     
     mean = mc_normspecflat(mean,edgecoeffs,xranges,flatinfo.slith_arc, $
                            flatinfo.norm_nxg,flatinfo.norm_nyg, $
                            flatinfo.oversamp,flatinfo.ybuffer,$
                            RMS=rms,IVAR=var,OVAR=ovar, $
                            /UPDATE,WIDGET_ID=state.w.xspextool_base, $
                            MODEL=model,CANCEL=cancel)
     if cancel then return    

  endif else ovar = var

;  Write the results to disk

  avehdr = mc_avehdrs(hdrinfo,CANCEL=cancel)
  if cancel then return
  
  history='This flat was created by scaling the files '+$
          strjoin(hdrinfo[*].vals.FILENAME, ', ')+ $
          ' to a common median flux value and then median combining ' + $
          'the scaled images.  The variance is given by (MAD^2/nimages) ' + $
          'where MAD is the median absolute deviation and nimages is the ' + $
          'number of input images.  The zeroth bit of pixels generated ' + $
          'from data with values greater than LINCORMX are set.  User ' + $
          'selected keywords are from the first frame in the series.'
  
  avehdr.vals.filename = file_basename(ofile)

  mc_writeflat,mc_unrotate(mean,flatinfo.rotation),$
               mc_unrotate(ovar,flatinfo.rotation),$
               mc_unrotate(byte(bitmask),flatinfo.rotation),$
               avehdr,flatinfo.rotation,fix(flatinfo.orders),edgecoeffs, $
               xranges,flatinfo.ps,flatinfo.slith_pix,flatinfo.slith_arc, $
               slitw_pix,slitw_arc,mode,resolvingpower, $
               state.w.version,ofile,LINCORMAX=state.r.lincormax, $
               RMS=rms,HISTORY=history,CANCEL=cancel
  if cancel then return

  xspextool_message,'Wrote flat to '+ofile

;  Update the Wavecal Box
  
  widget_control, w.flatfield_fld[1], SET_VALUE=strtrim(file_basename(ofile),2)
  
;  Display if requested.

  if keyword_set(DISPLAY) then begin
     
     mc_readflat,ofile,image,ncols,nrows,modename,ps,slith_pix,slith_arc,$
                 slitw_pix,slitw_arc,rp,norders,orders,edgecoeffs,xranges, $
                 rms,rotation,edgedeg,FLAGS=flags,OMASK=omask,CANCEL=cancel

     a = obj_new('mcoplotedge')
     a->set,xranges,edgecoeffs,ORDERS=orders

     ximgtool,ofile,EXT=1,ZRANGE=[0.9,1.1], $
              GROUP_LEADER=state.w.xspextool_base, $
              BUFFER=1,ROTATION=rotation,OPLOT=a,POSITION=[1,0], $
              STDIMAGE=state.r.stdimage,$
              BITMASK={bitmask:flags*(omask gt 0),plot1:[0,2]}, $
              PLOTWINSIZE=state.r.plotwinsize,/ZTOFIT,/CLEARALLFRAMES

     
     
  endif

  xspextool_message,'Task complete.',/WINONLY

  
  out:

end
;
;=============================================================================
;
pro ishellcals2dxd_wavecal,w,arcinfo,CANCEL=cancel

  cancel = 0
  
  common xspextool_state

  widget_control, /HOURGLASS

;  Figure out whether we are going to use arc or sky lines
  
  case w.usermode of
     
     'J0': type = 'arc'
     'J1': type = 'arc'
     'J2': type = 'arc'
     'J3': type = 'arc'
     'H1': type = 'arc'
     'H2': type = 'arc'
     'H3': type = 'arc'
     'K1': type = 'arc'
     'K2': type = 'arc'
     'Kgas': type = 'arc'     
     'K3': type = 'arc'
     'L1': type = 'arc'
     'L2': type = 'sky'
     'L3': type = 'sky'
     'Lp1': type = 'sky'
     'Lp2': type = 'sky'
     'Lp3': type = 'sky'
     'Lp4': type = 'sky'
     'M1': type = 'sky'
     'M2': type = 'sky'

  endcase

;  Get user innputs and check
  
  wconame = mc_cfld(w.wavecaloname_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  
  index    = (state.r.filereadmode eq 'Index') ? 1:0
  filename = (state.r.filereadmode eq 'Filename') ? 1:0
  
  if index then prefix = mc_cfld(state.w.iprefix_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return

  if type eq 'arc' then begin
     
     files = mc_cfld(w.arcimages_fld,7,CANCEL=cancel)
     if cancel then return
     field = 'Arc Images'
     
  endif else begin

     files = mc_cfld(w.skyimages_fld,7,CANCEL=cancel)
     if cancel then return
     field = 'Sky Images'
     
  endelse 

  if index then begin

;  Check for duplicate files
     
     result = mc_checkdups(state.r.datapath,files,state.r.nint, $
                           WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
     if cancel then return
     
  endif
  
;  Now read flat and cross check user obsmode and flat obsmode

  flatname = mc_cfld(w.flatfield_fld,7,/EMPTY,CANCEL=cancel)
  if cancel then return
  fullflatname = mc_cfile(state.r.calpath+flatname, $
                          WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
  if cancel then return  
  
  mc_readflat,fullflatname,flat,ncols,nrows,obsmode,ps,slith_pix,slith_arc, $
              slitw_pix,slitw_arc,rp,flatnorders,flatorders,flatedgecoeffs, $
              flatxranges,rms,rotation,edgedeg,OMASK=omask, $
              FLAGS=flatflags,CANCEL=cancel
  if cancel then return
  
  if w.usermode ne obsmode then begin
     
     message = 'User mode does not equal flat field mode.'
     result = dialog_message(message,/ERROR,$
                             DIALOG_PARENT=state.w.xspextool_base)
     return
     
  endif

;  Create the edge-plot object for later use with ximgtool
  
  flatedges = obj_new('mcoplotedge')
  flatedges->set,flatxranges,flatedgecoeffs,ORDERS=flatorders
  
;  goto, start
  
;  Now create the file names for the "arc" images
  
  files = mc_fsextract(files,INDEX=index,FILENAME=filename,NFILES=nfiles, $
                       CANCEL=cancel)
  if cancel then return

  files = mc_mkfullpath(state.r.datapath,files,INDEX=index,FILENAME=filename, $
                        NI=state.r.nint,PREFIX=prefix,SUFFIX='*.fits*', $
                        WIDGET_ID=state.w.xspextool_base, $
                        /EXIST,CANCEL=cancel)
  if cancel then return

;  Re order them to be ABABABAB.

  files = mc_reorder(files,MESSAGEINFO={id:state.w.xspextool_base,obj:field},$
                     CANCEL=cancel)
  if cancel then return

;  Now read the data in

  xspextool_message,'Loading data...'

  if type eq 'arc' then pair = 1
  
  call_procedure,'mc_readishellfits',files,data,hdrinfo,var, $
                 WIDGET_ID=state.w.xspextool_base,$
                 KEYWORDS=state.r.keywords,$
                 BITINFO={lincormax:state.r.lincormax,lincormaxbit:0},$
                 BITMASK=arcbitmask,ROTATE=rotation,$
                 PAIR=pair,NIMAGES=nimages,CANCEL=cancel
  if cancel then return

  if type eq 'sky' then begin
  
;  Now create the clean sky frames
     
     cube = fltarr(ncols,nrows,nfiles/2,/NOZERO)
     
     for i = 0,nfiles/2-1 do begin
        
        sky = data[*,*,i*2]+data[*,*,i*2+1] - abs(data[*,*,i*2]-data[*,*,i*2+1])
        cube[*,*,i] = sky
        
     endfor
     nimages = nfiles/2
     
  endif else cube = data
     
;  Combine images if necessary

  if nimages ne 1 then begin

     xspextool_message,'Combining images...'
     
     mc_meancomb,cube,arc,arcvar,DATAVAR=datavar,CANCEL=cancel
     if cancel then return

     bitmask = mc_combflagstack(arcbitmask,CANCEL=cancel)
     if cancel then return
     
  endif else begin
  
     arc    = reform(data)
     arcvar = reform(var)
     bitmask = arcbitmask
     
  endelse
  
  arc = temporary(arc)/flat
  arcvar = temporary(arcvar)/flat^2

;  Fix hot pixels and bad pixels

  xspextool_message,'Fixing bad pixels...'
  
  htpxmk = rotate(readfits(filepath('ishell_htpxmk.fits', $
                                    ROOT_DIR=state.r.packagepath, $
                                    SUBDIR='data'),/SILENT),rotation)
  
  bdpxmk = rotate(readfits(filepath('ishell_bdpxmk.fits', $
                                    ROOT_DIR=state.r.packagepath, $
                                    SUBDIR='data'),/SILENT),rotation)

  fixpix,arc,htpxmk*bdpxmk,odata,/SILENT
  arc = temporary(odata)

;  Now load the spectral cal file

  file = filepath(obsmode+'_wavecalinfo.fits',ROOT=state.r.packagepath, $
                  SUB='data')  
  wavecalinfo = mc_readwavecalinfo(file,CANCEL=cancel)
  if cancel then return

  xspextool_message,'Starting 1DXD wavelength calibration...'
    
;  Grab the edgecoefficients for the orders in the wavecal file
  
  match,flatorders,wavecalinfo.orders,idx

  wavecaledgecoeffs = flatedgecoeffs[*,*,idx]
  wavecalxranges = flatxranges[*,idx]
  
;  Create the edge-plot object for later use with ximgtool
  
  wavecaledges = obj_new('mcoplotedge')
  wavecaledges->set,wavecalinfo.xranges,wavecaledgecoeffs, $
                    ORDERS=wavecalinfo.orders

;  Create fake wavecal arrays to do a simple 1D sum extraction on the arc
  
  mc_simwavecal2d,ncols,nrows,wavecaledgecoeffs,wavecalxranges,slith_arc, $
                  ps,wavecal,spatcal,indices,CANCEL=cancel
  if cancel then return
  
;  Get the plot structures set up and display on ximgtool
  
  low_tracecoeffs = rebin([slith_arc/2.-wavecalinfo.extap/2.,0],2, $
                          wavecalinfo.norders)
  
  struc = mc_tracetoxy(omask,wavecal,spatcal,low_tracecoeffs,1, $
                       wavecalinfo.orders,replicate(1,wavecalinfo.norders), $
                       CANCEL=cancel)
  if cancel then return

  c = obj_new('mcplotap')
  c->set,struc,3

  hi_tracecoeffs = rebin([slith_arc/2.+wavecalinfo.extap/2.,0],2, $
                         wavecalinfo.norders)
  
  struc = mc_tracetoxy(omask,wavecal,spatcal,hi_tracecoeffs,1, $
                       wavecalinfo.orders,replicate(1,wavecalinfo.norders), $
                       CANCEL=cancel)
  if cancel then return

  d = obj_new('mcplotap')
  d->set,struc,3

  oplot = [c,d]
   
;  Display the arc
  
  ximgtool,arc,GROUP_LEADER=state.w.xspextool_base,STDIMAGE=state.r.stdimage, $
           PLOTWINSIZE=state.r.plotwinsize,BUFFER=1,POSITION=[1,0], $
           BITMASK={mask:bitmask,plot1:[0,2]},/ZTOFIT,WID=wid,PANNER=0,MAG=0,$
           OPLOT=[wavecaledges,oplot],/CLEARALLFRAMES

;  Extract the arc down the middle of the slit

  xspextool_message,'Extracting spectra...'
  
  aprange = [-wavecalinfo.extap/2.,wavecalinfo.extap/2.]+slith_arc/2.

  arcspec = mc_sumextspec(arc,arcvar,omask,wavecalinfo.orders,wavecal,spatcal, $
                          [+1],APRANGES=rebin(aprange,2,wavecalinfo.norders), $
                          /UPDATE,WIDGET_ID=state.w.xspextool_base,$
                          CANCEL=cancel)
  if cancel then return

  xspextool_message,'Determing offset from disk spectrum...'
  
;  Get setup for the cross correlation 
  
  wanchor = wavecalinfo.xcorspec[*,0]
  fanchor = wavecalinfo.xcorspec[*,1]

;  Figure out which order you are using for cross correlation

  zorder = where(wavecalinfo.orders eq wavecalinfo.xcororder)  
  warc = (arcspec.(zorder))[*,0]
  farc = (arcspec.(zorder))[*,1]
  
;  Do the cross correlation

  if state.r.plotautoxcorr[0] then begin

     xmc_corspec,wanchor,fanchor,warc,farc,offset, $
                 TITLE='Order '+string((wavecalinfo.orders)[zorder], $
                                       FORMAT='(I3.3)')+' Spectra', $
                 CANCEL=cancel
     if cancel then return
     
  endif else begin

     xmc_corspec,wanchor,fanchor,warc,farc,offset,/JUSTFIT,CANCEL=cancel
     if cancel then return
     
  endelse

  xspextool_message,'Cross-correlation offset = '+strtrim(offset,2)+' pixels.'

;  Get the line list
  
  file = filepath(wavecalinfo.linelist,ROOT=state.r.packagepath,SUB='data')
  lineinfo = mc_readlinelist(file,CANCEL=cancel)
  if cancel then return
  
;  Conver the window to microns
  
  lineinfo.wwin = lineinfo.wwin/1d4
  
;  Determine the guess positions
  
  lineinfo = mc_getlinexguess(reform((wavecalinfo.wspec)[*,0,*]), $
                              wavecalinfo.xranges,wavecalinfo.orders, $
                              lineinfo,CANCEL=cancel)
  if cancel then return

;  Add in the offset
  
  lineinfo.xguess = lineinfo.xguess+offset

;  Get QA stuff for the wavelength calibration call

  if state.r.mk1dlinefindplot then begin

     qalinefindplot = {mode:state.r.calpath+wconame,loglin:state.r.mkloglinplot}

  endif
  
;  Call the 1DXD wavelength calibration routine

  mc_wavecal1dxd,arcspec,wavecalinfo.orders,lineinfo,wavecalinfo.homeorder, $
                 wavecalinfo.dispdeg,wavecalinfo.ordrdeg, $
                 p2wcoeffs,p2wrms, $
                 QAPLOT={mode:state.r.calpath+wconame,ncols:ncols}, $
                 QALINEFINDPLOT=qalinefindplot, $
                 OLINEINFO=olineinfo,CANCEL=cancel
  if cancel then return
  lineinfo = olineinfo

;  save,lineinfo,FILENAME='lineinfo.sav'
;  return
  
; Report the results

  junk = where(lineinfo.fnd_goodbad eq 1,cntgood1)
  junk = where(lineinfo.fit_goodbad eq 1,cntgood2)
  
  xspextool_message, '1DXD Solution:  '+strtrim(cntgood1,2)+ $
                     ' lines used in calibration, '+ $
                     strtrim(cntgood1-cntgood2,2)+ $
                     ' lines were identified as bad.'
  
  xspextool_message,'1DXD wavelength calibration complete.',/SPACE

  xspextool_message,'Starting distortion map...'
  
  if ~w.docoeffs then begin

;  Now trace the lines to map out the distortion

     ximgtool,arc,GROUP_LEADER=state.w.xspextool_base, $
              STDIMAGE=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize, $
              BUFFER=1,POSITION=[1,0],BITMASK={mask:bitmask,plot1:[0,2]}, $
              /ZTOFIT,WID=wid,PANNER=0,MAGNIFIER=0,OPLOT=wavecaledges
     
;  Find the lines

     xspextool_message,'Identifying lines...'

     linexyinfo = mc_findlines2d(arc,wavecaledgecoeffs,wavecalxranges, $
                                 wavecalinfo.orders,lineinfo, $
                                 wavecalinfo.linedeg,wavecalinfo.fndystep, $
                                 wavecalinfo.fndysum, $
                                 IGOODBAD=lineinfo.fit_goodbad, $
                                 OGOODBAD=ogoodbad,WID=wid,CANCEL=cancel)
     if cancel then return

;  Report the results

     junk = where(lineinfo.fit_goodbad eq 1,cntgood1)
     junk = where(ogoodbad eq 1,cntgood2)

     xspextool_message, '2DXD Line Find:  '+strtrim(cntgood1,2)+ $
                        ' lines were searched for and '+strtrim(cntgood2,2)+ $
                        ' lines were found.'

;  Clip the lineinfo and linexyinfo structures so you are only dealing
;  with good lines

     z = where(ogoodbad eq 1)
     lineinfo = lineinfo[z]
     linexyinfo = mc_indxstruc(linexyinfo,z,CANCEL=cancel)
     if cancel then return
     
;  Fit the lines
     
     xspextool_message,'Fitting lines...'

     if state.r.mklinesummaryplot then begin

        linesumqaplot={ncols:state.r.ncols,nrows:state.r.nrows, $
                       prefix:state.r.calpath+wconame}

     endif

     if state.r.mk2dlinefitsplot then begin
        
        linefitqaplot = {img:arc,prefix:state.r.calpath+wconame, $
                         waves:lineinfo.swave,fwhms:lineinfo.fwhm,$
                         ids:lineinfo.id}
        
     endif
     
     lstruc = mc_fitlines2dxd(linexyinfo,lineinfo.order,wavecaledgecoeffs, $
                              wavecalxranges,wavecalinfo.orders, $
                              wavecalinfo.linedeg,LINEFITQAPLOT=linefitqaplot, $
                              LINESUMQAPLOT=linesumqaplot,CANCEL=cancel)
     if cancel then return
     
;  Fit line coefficients

     xspextool_message,'Fitting line coefficients...'

     coeff = [wavecalinfo.c1xdeg,wavecalinfo.c1ydeg]
     
     if wavecalinfo.linedeg eq 2 then coeff = [coeff,wavecalinfo.c2xdeg, $
                                               wavecalinfo.c2ydeg]

     mc_fitlinecoeffs2d,lstruc,coeff,c1coeffs,c2coeffs, $
                        QAPLOT={ncols:state.r.ncols,nrows:state.r.nrows, $
                                prefix:state.r.calpath+wconame,$
                                xrange:[min(wavecalinfo.xranges,MAX=max),max]},$
                        MODEINFO={edgecoeffs:wavecaledgecoeffs, $
                                  xranges:wavecalxranges, $
                                  orders:wavecalinfo.orders, $
                                  lorders:lineinfo.order},$
                        CANCEL=cancel
     if cancel then return
     
  endif else begin
     
  endelse
  
  xspextool_message,'Generating rectification indices...'

;  Create the interpolation indices

  coeffs = {c1coeffs:c1coeffs,c1xdeg:wavecalinfo.c1xdeg, $
            c1ydeg:wavecalinfo.c1ydeg}
  
  if wavecalinfo.linedeg eq 2 then begin
     
     coeffs = create_struct(coeffs,'c2coeffs',c2coeffs,'c2xdeg', $
                            wavecalinfo.c2xdeg,'c2ydeg',wavecalinfo.c2ydeg)
     
  endif

;  z = where(orders eq 164)
;  edgecoeffs = edgecoeffs[*,*,z]
;  orders = orders[z]
;  xranges = xranges[*,z]
;  norders =1
;
;  c1xdeg = wavecalinfo.c1xdeg
;  c1ydeg = wavecalinfo.c1ydeg
;  genystep = wavecalinfo.genystep
;  
;  save,FILENAME='save.sav',edgecoeffs,xranges,c1coeffs,c1xdeg,c1ydeg, $
;       slith_arc,ps,c2info,genystep,arc
;  return

  indices = mc_mkrectindcs2d(flatedgecoeffs,flatxranges,coeffs,slith_arc,ps, $
                             wavecalinfo.genystep,XGRIDS=xgrids,SGRID=sgrid, $
                             /UPDATE,WIDGET_ID=state.w.xspextool_base, $
                             CANCEL=cancel)
  if cancel then return

;  start:
;  restore, 'save.sav'
;
;
;  indices = mc_mkrectindcs(edgecoeffs,xranges,c1coeffs,c1xdeg, $
;                           c1ydeg,slith_arc,ps, $
;                           genystep,$
;                           C2COEFFSINFO=c2info,XGRIDS=xgrids, $
;                           SGRID=sgrid,/UPDATE, $
;                           WIDGET_ID=state.w.xspextool_base,CANCEL=cancel)
;  if cancel then return
;  
;  ix = (indices.(0))[*,*,0]
;  iy = (indices.(0))[*,*,1]
;
;  a = obj_new('mcoplot')
;  a->set,ix,iy,PSYM=1,COLOR=3,THICK=1
;
;  ximgtool,arc,OPLOT=[a,edges]
;  return
  
;  Now let's repackage them so they include the wgrid, and sgrid data all
;  in one.  Find the dispersions while you are at it.

  disp = fltarr(flatnorders,/NOZERO)
  xranges = intarr(2,flatnorders,/NOZERO)
  for i =0,flatnorders-1 do begin

;  Make the wgrid first
     
     ordrscl = total(flatorders[i]/float(wavecalinfo.homeorder))
     wgrid = mc_poly2d(xgrids.(i),replicate(flatorders[i], $
                                            n_elements(xgrids.(i))), $
                       wavecalinfo.dispdeg,wavecalinfo.ordrdeg,p2wcoeffs)/ $
             ordrscl

;     wgrid = mc_findlambdahome(wgrid,replicate(wavecalinfo.homeorder,cnt), $
;                               orders[i],30,wavecalinfo.wrange[0], $
;                               wavecalinfo.wrange[1],CANCEL=cancel)
;     if cancel then return

;  Now create and fill in a new array containing indices, xgrid,
;  wgrid, and sgrid
     
     s = size((indices.(i))[*,*,0],/DIMEN)
     nindices = make_array(s[0]+1,s[1]+1,2,/DOUBLE,VALUE=!values.f_nan)
     
     nindices[1:*,0,0] = wgrid
     nindices[1:*,0,1] = wgrid
     nindices[0,1:*,0] = sgrid
     nindices[0,1:*,1] = sgrid
     nindices[1:*,1:*,0] = (indices.(i))[*,*,0]
     nindices[1:*,1:*,1] = (indices.(i))[*,*,1]

;  Compute the dispersion
     
     coeff = mc_polyfit1d(xgrids.(i),wgrid,1,/SILENT)
     
     disp[i] = coeff[1]

;  Get the xranges

     xranges[*,i] = [min(xgrids.(i),MAX=max),max]
     
;  Store the results     

     tag = string(i)
     findices = (i eq 0) ? create_struct(tag,nindices): $
                create_struct(findices,tag,nindices)
     
  endfor

;  Generate the wavecal and spatcal images for use with ximgtool

  mc_mkwavecalimgs2d,omask,flatorders,findices,wavecal,spatcal,CANCEL=cancel
  if cancel then return

  xspextool_message,'Distortion map complete.',/SPACE

;  Write the results to disk
  
  xspextool_message,'Generating Wavelength Calibrarion File...'
  
;  Write the resulting wave and spat cal images to disk
  
  mwrfits,input,state.r.calpath+wconame+'.fits',nhdr,/CREATE
  sxaddpar,nhdr,'IRAFNAME',wconame+'.fits',' Filename'
  sxaddpar,nhdr,'FLATNAME', flatname,' Associated flat field frame'
  fxaddpar,nhdr,'ORDERS',strjoin(strcompress(flatorders,/re),','), $
           ' Orders identified'
  fxaddpar,nhdr,'NORDERS',flatnorders, ' Number of orders identified'
  sxaddpar,nhdr,'EXTTYPE','2D', ' Extraction type'
  sxaddpar,nhdr,'WCTYPE','2DXD',' Wavelength calibration type'
  sxaddpar,nhdr,'WAVEFMT', wavecalinfo.wavefmt,' Wavelength format statement'
  sxaddpar,nhdr,'SPATFMT', wavecalinfo.spatfmt,' Angle format statement'
  
;  Write the RMS of the fit
  
  sxaddpar,nhdr,'RMS',string(p2wrms*1e4,FORMAT='(f5.3)'), $
           ' RMS of 1DXD wavecal fit in Angstroms'

;  Now add the xranges
  
  for i = 0, flatnorders-1 do begin
     
     name    = 'OR'+string(flatorders[i],FORMAT='(i3.3)')+'_XR'
     comment = ' Extraction range for order '+ $
               string(flatorders[i],FORMAT='(i3.3)')
     sxaddpar,nhdr,name,strjoin(strtrim(flatxranges[*,i],2),',',/SINGLE),comment
     
  endfor
  
;  Write the dispersions
    
  for j = 0, flatnorders-1 do begin
        
        name = 'DISPO'+string(flatorders[j],format='(i3.3)')
        comment = ' Dispersion (um pix-1) for order '+ $
                  string(flatorders[j],FORMAT='(i3.3)')
        sxaddpar,nhdr,name,disp[j],comment
        
     endfor

  mwrfits,input,state.r.calpath+wconame+'.fits',nhdr,/CREATE

;  Write wavelength calibration extension

  writefits,state.r.calpath+wconame+'.fits',mc_unrotate(wavecal,5),/APPEND

;  Write spatial calibration to second extension

  writefits,state.r.calpath+wconame+'.fits',mc_unrotate(spatcal,5),/APPEND

;  Now write the indices

  for i = 0,flatnorders-1 do begin
     
     writefits,state.r.calpath+wconame+'.fits',findices.(i),/APPEND

  endfor

  !x.thick=1
  !y.thick=1
  !p.thick=1

  speccoords = {ordr:omask,$
                wave:wavecal,wunits:'um',wfmt:wavecalinfo.wavefmt, $
                spat:spatcal,sunits:'"',sfmt:wavecalinfo.spatfmt}
  
  ximgtool,arc,GROUP_LEADER=state.w.xspextool_base,$
           STDIMAGE=state.r.stdimage,PLOTWINSIZE=state.r.plotwinsize, $
           BUFFER=1,OPLOT=flatedges,/LOCK,/ZTOFIT,SPECCOORDS=speccoords

  xspextool_message,'Wavelength calibration complete.',/SPACE

  out:
  
end
;
;=============================================================================
;
;------------------------------ Event Handler -------------------------------
;
;=============================================================================
;
pro ishellcals2dxd_event,event

  common xspextool_state

  widget_control, event.handler, GET_UVALUE = w, /NO_COPY
  widget_control, event.id,  GET_UVALUE = uvalue

  case uvalue of 

     'Arc Combine Mode': w.arccombmode=event.value
     
     'Arc Images Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.datapath,$
                              FILTER=['*.fits;*.fits.gz'],/FIX_FILTER,$
                              /MULTIPLE_FILES)
        
        if path[0] ne '' then $
           widget_control,w.arcimages_fld[1],$
                          SET_VALUE=strjoin(file_basename(path),',',/SINGLE)
        
     end

     'Clear Table': widget_control, w.table, SET_VALUE=strarr(3,w.nrows)
     
     'Flat Field Images Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.datapath,$
                              FILTER=['*.fits;*.fits.gz'],/FIX_FILTER,$
                              /MULTIPLE_FILES)
        
        if path[0] ne '' then $
           widget_control,w.flats_fld[1],$
                          SET_VALUE=strjoin(file_basename(path),',',/SINGLE)

     end

     'Full Flat Name Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.calpath,$
                              FILTER='*.fits',/FIX_FILTER)
        
        if path ne '' then widget_control,w.flatfield_fld[1], $
                                          SET_VALUE=file_basename(path)

        mc_setfocus,w.flatfield_fld
                
     end

     'Line Coeffs Mode': w.docoeffs = event.value
     
     'Make Flat Field Button': begin

        ishellcals2dxd_mkflat,w,/DISPLAY,CANCEL=cancel
        if cancel then goto, out
        xspextool_message,'Task complete.',/WINONLY

     end

     'Mode': begin

        widget_control,w.arc_base,MAP=0
        widget_control,w.sky_base,MAP=0
        case (w.modes)[event.index] of

           'J0': widget_control,w.arc_base,MAP=1
           'J1': widget_control,w.arc_base,MAP=1
           'J2': widget_control,w.arc_base,MAP=1
           'J3': widget_control,w.arc_base,MAP=1
           'H1': widget_control,w.arc_base,MAP=1
           'H2': widget_control,w.arc_base,MAP=1
           'H3': widget_control,w.arc_base,MAP=1
           'K1': widget_control,w.arc_base,MAP=1
           'K2': widget_control,w.arc_base,MAP=1
           'Kgas': widget_control,w.arc_base,MAP=1
           'K3': widget_control,w.arc_base,MAP=1
           'L1': widget_control,w.arc_base,MAP=1
           'L2': widget_control,w.sky_base,MAP=1
           'L3': widget_control,w.sky_base,MAP=1
           'L4': widget_control,w.sky_base,MAP=1
           'Lp1': widget_control,w.sky_base,MAP=1
           'Lp2': widget_control,w.sky_base,MAP=1
           'Lp3': widget_control,w.sky_base,MAP=1
           'Lp4': widget_control,w.sky_base,MAP=1
           'M1': widget_control,w.sky_base,MAP=1
           'M2': widget_control,w.sky_base,MAP=1

        endcase

        w.usermode = (w.modes)[event.index]
        
     end

     'Sky Images Button': begin
        
        path= dialog_pickfile(DIALOG_PARENT=state.w.xspextool_base,$
                              /MUST_EXIST,PATH=state.r.datapath,$
                              FILTER=['*.fits;*.fits.gz'],/FIX_FILTER,$
                              /MULTIPLE_FILES)
        
        if path[0] ne '' then $
           widget_control,w.skyimages_fld[1],$
                          SET_VALUE=strjoin(file_basename(path),',',/SINGLE)
        
     end

     
     'Sky Mode': w.skypairsub = event.value
     
     'Wavelength Calibrate': begin

        ishellcals2dxd_wavecal,w,CANCEL=cancel
        if cancel then goto, out
        xspextool_message,'Task complete.',/WINONLY

     end

     else:

  endcase

  out:
  
  widget_control, event.handler, SET_UVALUE=w, /NO_COPY

end
;
;============================================================================
;
;------------------------------ Main Program -------------------------------
;
;============================================================================
;
pro mc_ishellcals2dxd,parent,FAB=fab,ENG=eng,CANCEL=cancel

  common xspextool_state

  FAB=1
  
  w = {arc_base:0L,$
       arcimages_fld:[0L,0L],$
       arccombmode:'A',$
       arconame_fld:[0L,0L],$
       coeffs_bg:0L,$
       docoeffs:0L,$
       eng:keyword_set(ENG),$
       flatoname_fld:[0L,0L],$
       flatfield_fld:[0L,0L],$
       flats_fld:[0L,0L],$
       sky_base:0L,$
       skyimages_fld:[0L,0L],$
       skymode_bg:0L,$
       skypairsub:1,$
       table:0L,$
       modes:['J0','J1','J2','J3','H1','H2','H3','K1','K2','Kgas','K3','L1', $
              'L2','L3','L4','Lp1','Lp2','Lp3','Lp4','M1','M2'],$
       usermode:'J0',$
       nrows:8,$
       wavecaloname_fld:[0L,0L]}

  mc_getfonts,buttonfont,textfont

  row_base = widget_base(parent,$
                         /ROW)
  
  col1_base = widget_base(row_base,$
                          /COLUMN,$
                          FRAME=2)
  
     label = widget_label(col1_base,$
                          VALUE='1. Flat Field',$
                          FONT=buttonfont,$
                          /ALIGN_LEFT)
     
     row = widget_base(col1_base,$
                       /ROW,$
                       /BASE_ALIGN_CENTER)
     
        button = widget_button(row,$
                               FONT=buttonfont,$
                               VALUE='Raw Flat Images',$
                               UVALUE='Flat Field Images Button')        
        fld = coyote_field2(row,$
                            LABELFONT=buttonfont,$
                            FIELDFONT=textfont,$
                            TITLE=':',$
                            UVALUE='Flat Field Images Field',$
                            XSIZE=10,$
;                            VALUE='25-29',$
                            TEXTID=textid) 
        w.flats_fld = [fld,textid]

     fld = coyote_field2(col1_base,$
                         LABELFONT=buttonfont,$
                         FIELDFONT=textfont,$
                         TITLE='Flat Output Name:',$
                         UVALUE='Flat Output Name',$
                         XSIZE=15,$
;                         VALUE='ftest',$
                         TEXTID=textid) 
     w.flatoname_fld = [fld,textid]
     
     button = widget_button(col1_base,$
                            FONT=buttonfont,$
                            VALUE='Generate Flat Field',$
                            UVALUE='Make Flat Field Button')
     
  col2_base = widget_base(row_base,$
                          /COLUMN,$
                          FRAME=2)
  
     label = widget_label(col2_base,$
                          Value='2. Wavelength Calibration',$
                          FONT=buttonfont,$
                          /ALIGN_LEFT)

     model_dl = widget_droplist(col2_base,$
                                FONT=buttonfont,$
                                TITLE='Mode:',$
                                VALUE=w.modes,$
                                UVALUE='Mode')     

     base = widget_base(col2_base)

        w.arc_base = widget_base(base,$
                                 /COLUMN,$
                                 /BASE_ALIGN_LEFT,$
                                 MAP=1)   
        
           row = widget_base(w.arc_base,$
                             /ROW,$
                             /BASE_ALIGN_CENTER)
           
              button = widget_button(row,$
                                     FONT=buttonfont,$
                                     VALUE='Arc Images',$
                                     UVALUE='Arc Images Button')
              
              fld = coyote_field2(row,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE=':',$
                                  UVALUE='Arc Images Field',$
                                  XSIZE=10,$
 ;                                 VALUE='11-14',$
                                  TEXTID=textid) 
              w.arcimages_fld = [fld,textid]

        w.sky_base = widget_base(base,$
                                 /COLUMN,$
                                 /BASE_ALIGN_LEFT,$
                                 MAP=0)   

           row = widget_base(w.sky_base,$
                             /ROW,$
                             /BASE_ALIGN_CENTER)

           w.skymode_bg = cw_bgroup(row,$
                                    FONT=buttonfont,$
                                    ['A','A-B'],$
                                    /ROW,$
                                    /NO_RELEASE,$
                                    /EXCLUSIVE,$
                                    /RETURN_INDEX,$
                                    SET_VALUE=1,$
                                    UVALUE='Sky Mode')

           
           
              button = widget_button(row,$
                                     FONT=buttonfont,$
                                     VALUE='Sky Images',$
                                     UVALUE='Sky Images Button')
              
              fld = coyote_field2(row,$
                                  LABELFONT=buttonfont,$
                                  FIELDFONT=textfont,$
                                  TITLE=':',$
                                  UVALUE='Sky Images Field',$
                                  XSIZE=25,$
;                                  VALUE='1-6',$
                                  TEXTID=textid) 
              w.skyimages_fld = [fld,textid]
        
     row = widget_base(col2_base,$
                       /ROW,$
                       /BASE_ALIGN_CENTER)
        
        button = widget_button(row,$
                               FONT=buttonfont,$
                               VALUE='Full Flat Name',$
                               UVALUE='Full Flat Name Button')
        
        fld = coyote_field2(row,$
                            LABELFONT=buttonfont,$
                            FIELDFONT=textfont,$
                            TITLE=':',$
;                            VALUE='flat15-19.fits',$
                            UVALUE='Full Flat Name',$
                            XSIZE=20,$
                            TEXTID=textID)
        w.flatfield_fld = [fld,textID]
        
;     w.coeffs_bg = cw_bgroup(col2_base,$
;                             FONT=buttonfont,$
;                          ['Generate Line Coeffs','Used Stored Line Coeffs'],$
;                                   /ROW,$
;                                   /NO_RELEASE,$
;                                   /EXCLUSIVE,$
;                                   /RETURN_INDEX,$
;                                   UVALUE='Line Coeffs Mode')
;
;     widget_control,w.coeffs_bg,SET_VALUE=w.docoeffs
     
     fld = coyote_field2(col2_base,$
                         LABELFONT=buttonfont,$
                         FIELDFONT=textfont,$
                         TITLE='Wavecal Output Name:',$
                         UVALUE='Wavecal Output Name',$
                         XSIZE=20,$
;                         VALUE='wavecal11-14',$
                         TEXTID=textid) 
     w.wavecaloname_fld = [fld,textid]
     
     button = widget_button(col2_base,$
                            FONT=buttonfont,$
                            VALUE='Generate Wavelength Calibration',$
                            UVALUE='Wavelength Calibrate')
     
; Start the Event Loop. This will be a non-blocking program.

  XManager, 'xmc_ishellcals2dxd', $
            row_base, $
            EVENT_HANDLER='ishellcals2dxd_event',$
            /NO_BLOCK

; Put state variable into the user value of the top level base.

  widget_control, row_base, SET_UVALUE=w, /NO_COPY


end
