;+
; NAME:
;     mc_wavecal1dxd
;
; PURPOSE:
;     To perform a wavelength calibration on XD spectra
;
; CALLING SEQUENCE:
;     mc_wavecal1dxd,spectra,orders,lineinfo,homeorder,dispdeg,ordrdeg, $
;                    p2wcoeffs,p2wrms,OLINEINFO=olineinfo,SIIOR=siior, $
;                    QAPLOT=qaplot,QALINEFINDPLOT=qalinefindplot, $
;                    PLOTFIND=plotfind,CANCEL=cancel
;
; INPUTS:
;     spectra   - A structure with norders tags.  Each tag is the
;                 spectrum for a given order:
;                 wave=(spectra.(i))[*,0]
;                 flux=(spectra.(i))[*,1]
;     orders    - A [norders] array of order numbers
;     lineinfo  - A structure with the following tags:
;                 lineinfo.order   = order numbers (integer)
;                 lineinfo.swave   = wavelengths (string)
;                 lineinfo.id      = line IDs (string)
;                 lineinfo.wwin    = fit window (in units of swave)
;                 lineinfo.fittype = fit type (0=gaussian, 1=Lotentzian)
;                 lineinfo.nterms  = Number of terms (3=basic,
;                                    4=basic+constant,5=basic+line) 
;                 lineinfo.xguess  = x positions of lines in wspec.
;                 lineinfo.xwin    = The fit window in units of pixels.
;     homeorder - The order number to scale the wavelengths of lines
;                 by (order/homeorder) in order to perform the 1DXD fit.  
;     dispdeg   - The 2D polynomial fit degree in the dispersion direction.
;     ordrdeg   - The 2D polynomial fit degree in the order direction.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     OLINEINFO      - A structure with the following tags:
;                      lineinfo.order = order numbers (integer)
;                      lineinfo.swave = wavelengths (string)
;                      lineinfo.id = line IDs (string)
;                      lineinfo.wwin = fit window (in units of swave)
;                      lineinfo.fittype = fit type (0=gaussian, 1=Lotentzian)
;                      lineinfo.nterms = Number of terms (3=basic,
;                                        4=basic+constant,5=basic+line) 
;                      lineinfo.xguess = x positions of lines in wspec.
;                      lineinfo.xwin = The fit window in units of pixels.
;                      lineinfo.xpos = the fitted position
;                      lineinfo.fwhm = the fwhm of the fit
;                      lineinfo.inten = the intensity of the line
;                      lineinfo.fnd_goodbad = The goodbad array for the
;                      line search.
;                      lineinfo.fit_goodbad = The goodbad array for the
;                      1dxd fit.
;     SIIOR          - Not implemented.
;     QAPLOT         - If given, the residuals QA plot with be generated.
;
;                      A structure with the following tags:
;                      qaplot.mode=A string giving the mode name.
;                      qaplot.ncols=The number of columns on the
;                      array.
;                      qaplot.fullrange=Set to plot the full range of
;                      residuals (including bad points).  Otherwise,
;                      the yrange is set to +-5 sigma of the good
;                      points.
;     QALINEFINDPLOT - If given, the line finding QA plot with be
;                      generated.
;                      qaplot.mode=A string giving the mode name.
;                      qaplot.loglin=Set to plot the hybrid
;                      logarithm/linear plot.
;     PLOTFIND       - Set to plot the line finding interactively.
;     CANCEL         - Set on return if there is a problem.
;
; OUTPUTS:
;     p2wcoeffs - 2D [dispdeg+1 * ordrdeg+1] 2D polynomial
;                 coefficients.
;     p2wrms    - The RMS of the fit of the good points.
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
;     Identifies positions of the lines given in lineinfo, and then
;     performs the 2DXD fit for the solution. 
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-09-01 - Written by M. Cushing, University of Toledo 
;-
pro mc_wavecal1dxd,spectra,orders,lineinfo,homeorder,dispdeg,ordrdeg, $
                   p2wcoeffs,p2wrms,OLINEINFO=olineinfo,SIIOR=siior, $
                   QAPLOT=qaplot,QALINEFINDPLOT=qalinefindplot, $
                   PLOTFIND=plotfind,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() lt 6 then begin
     
     print, 'Syntax - mc_wavecal1dxd,spectra,orders,lineinfo,homeorder,$'
     print, '                        dispdeg,ordrdeg,p2wcoeffs,p2wrms,$'
     print, '                        OLINEINFO=olineinfo,SIIOR=siior,$'
     print, '                        QAPLOT=qaplot,$'
     print, '                        QALINEFINDPLOT=qalinefindplot,$'
     print, '                        PLOTFIND=plotfind,CANCEL=cancel'
     cancel = 1
     return
     
  endif
  cancel = mc_cpar('mc_wavecal1dxd',spectra,1,'spectra',8)
  if cancel then return
  cancel = mc_cpar('mc_wavecal1dxd',orders,2,'Orders',[2,3,4,5],[0,1])
  if cancel then return
  cancel = mc_cpar('mc_wavecal1dxd',lineinfo,3,'Lineinfo',8)
  if cancel then return
  cancel = mc_cpar('mc_wavecal1dxd',homeorder,4,'Homeorder',[2,3],0)
  if cancel then return  
  cancel = mc_cpar('mc_wavecal1dxd',dispdeg,5,'Dispdeg',[2,3],0)
  if cancel then return
  cancel = mc_cpar('mc_wavecal1dxd',ordrdeg,6,'Ordrdeg',[2,3],0)
  if cancel then return    

;  Get info and get started.
  
  norders = n_elements(orders)
  nlines = n_elements(lineinfo[*].swave)

;  Figure out whether the QA RANGE should be 

  fullrange = (keyword_set(QAPLOT) eq 1 and n_tags(qaplot) eq 3) ? $
              qaplot.fullrange:0

;  Find the lines
  
  lineinfo = mc_findlines1dxd(spectra,orders,lineinfo,2,PLOTFIND=plotfind, $
                              QAPLOT=qalinefindplot,CANCEL=cancel)
  if cancel then return
     
;  Deal with the index of refraction issue if necessary
     
;     if keyword_set(SIIOR) then begin
;
;        tmp = mc_findlambdahome(lwaves[zlines],lorders[zlines], $
;                                homeorder,30,siior[0],siior[1],CANCEL=cancel)
;        if cancel then return
;        lswave = [lswave,tmp]
;
;     endif else lswave=[lswave,lwaves[zlines]*lorders[zlines]/float(homeorder)]
          
;  Now perform the 1DXD wavelength calibration

  sclwave = double(lineinfo.swave)*float(lineinfo.order)/float(homeorder)
  p2wcoeffs = mc_robustpoly2d(lineinfo.xpos,lineinfo.order,sclwave,dispdeg, $
                              ordrdeg,3,0.01,/SILENT,RMS=p2wrms, $
                              IGOODBAD=lineinfo.fnd_goodbad, $
                              OGOODBAD=p2wogoodbad,CANCEL=cancel)
  if cancel then return
  
;  Add the goodbad array into the lineinfo structure
  
  olineinfo = lineinfo[0]
  olineinfo = create_struct(olineinfo,'fit_goodbad',0)
  olineinfo = replicate(olineinfo,nlines)
  struct_assign, lineinfo, olineinfo
  olineinfo.fit_goodbad = p2wogoodbad
  
;  Now do QA plot if requested

  if keyword_set(QAPLOT) then begin

     zfit = mc_poly2d(olineinfo.xpos,olineinfo.order,dispdeg,ordrdeg,p2wcoeffs)
     resid = (sclwave-zfit)*10000.

;  Find the statistics of the residuals
     
     z = where(p2wogoodbad eq 1)
     m = moment(resid[z])
     if ~fullrange then yrange = [m[0]-5*sqrt(m[1]),m[0]+5*sqrt(m[1])]
     
;  Find the bad/good points

     zfitbad = where(olineinfo.fit_goodbad eq 0 and $
                     olineinfo.fnd_goodbad eq 1,fitbadcnt)
     zfndbad = where(olineinfo.fnd_goodbad eq 0,fndbadcnt,$
                     NCOMP=fndgoodcnt)
     
     mc_ldcolortempct,norders

     !x.thick=3
     !y.thick=3
     !p.thick=3
     
     mc_setpsdev,qaplot.mode+'_1DXDResiduals.eps',7.5,10,FONT=14
     
     positions = mc_gridplotpos([0.15,0.075,0.97,0.95],[1,2],0.08,/COL)
     
     plot,olineinfo.order,resid,PSYM=8,POSITION=positions[*,0],FONT=0,/NODATA,$
          XRANGE=[min(olineinfo.order,MAX=max)-0.5,max+0.5],/XSTY,/YSTY,$
          XTITLE='Order Number',YTITLE='Data-Model (Angstroms)',YRANGE=yrange,$
          LINESTYLE=1,XMINOR=1
     
     plotsym,0,0.5,/FILL
     for i = 0,norders-1 do begin
        
        z = where(olineinfo.order eq orders[i],cnt)
        if cnt ne 0 then begin
           
           plotsym,0,0.9,/FILL
           oplot,[olineinfo[z].order],resid[z],PSYM=8
           plotsym,0,0.7,/FILL
           oplot,[olineinfo[z].order],resid[z],COLOR=2+i,PSYM=8
           
        endif
        
     endfor     
     
     if fitbadcnt ne 0 then oplot,[olineinfo[zfitbad].order],resid[zfitbad], $
                                  PSYM=6,SYMSIZE=1.5

     if fndbadcnt ne 0 then oplot,[olineinfo[zfndbad].order],resid[zfndbad], $
                                  PSYM=4,SYMSIZE=1.5     
     
     plots,!x.crange,[m[0],m[0]],LINESTYLE=2
     plots,!x.crange,[m[0],m[0]]-sqrt(m[1]),LINESTYLE=1
     plots,!x.crange,[m[0],m[0]]+sqrt(m[1]),LINESTYLE=1
     
     plot,olineinfo.xpos,resid,PSYM=8,POSITION=positions[*,1],FONT=0,/NODATA, $
          /NOERASE,/XSTY,/YSTY,XTITLE='Column (pixels)',$
          YTITLE='Data-Model (Angstroms)',YRANGE=yrange, $
          XRANGE=[0,qaplot.ncols-1]
     
     for i = 0,norders-1 do begin
        
        z = where(olineinfo.order eq orders[i],cnt)
        
        if cnt ne 0 then begin
           
           plotsym,0,0.9,/FILL
           oplot,[olineinfo[z].xpos],resid[z],PSYM=8
           plotsym,0,0.7,/FILL
           oplot,[olineinfo[z].xpos],resid[z],COLOR=2+i,PSYM=8
           
        endif
        
     endfor
     
     plotsym,0,1

     if fitbadcnt ne 0 then oplot,[olineinfo[zfitbad].xpos],resid[zfitbad], $
                                  PSYM=6,SYMSIZE=1.5

     if fndbadcnt ne 0 then oplot,[olineinfo[zfndbad].xpos],resid[zfndbad], $
                                  PSYM=4,SYMSIZE=1.5     
     
     plots,!x.crange,[m[0],m[0]],LINESTYLE=2
     plots,!x.crange,[m[0],m[0]]-sqrt(m[1]),LINESTYLE=1
     plots,!x.crange,[m[0],m[0]]+sqrt(m[1]),LINESTYLE=1

     offset = 0.02
     xyouts,positions[0,0]+0.025,positions[3,0]-offset,'!9l!X Deg='+ $
            strtrim(fix(dispdeg),2), $
            FONT=0,ALIGNMENT=0,/NORM,CHARSIZE=0.9
     
     xyouts,positions[0,0]+0.025,positions[3,0]-2*offset,'Order Deg='+ $
            strtrim(fix(ordrdeg),2), $
            FONT=0,ALIGNMENT=0,/NORM,CHARSIZE=0.9
     
     xyouts,positions[0,0]+0.025,positions[3,0]-3*offset,'RMS = '+ $
            string(sqrt(m[1]),FORMAT='(f5.3)')+' '+string(197B),$
            FONT=0,ALIGNMENT=0,/NORM,CHARSIZE=0.9

     xyouts,positions[0,0]+0.025,positions[3,0]-4*offset,'n!Dtot!N= '+ $
            strtrim(string(fndgoodcnt,FORMAT='(I4)'),2)+', n!Dbad!N = '+ $
            strtrim(string(fitbadcnt,FORMAT='(I2)'),2),$
            FONT=0,ALIGNMENT=0,/NORM,CHARSIZE=0.9
          
     mc_setpsdev,/CLOSE,/CONVERT,/ERASE

     !x.thick=1
     !y.thick=1
     !p.thick=1
     
  endif
     

end
