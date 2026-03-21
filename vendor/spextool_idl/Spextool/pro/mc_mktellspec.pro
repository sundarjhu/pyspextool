;+
; NAME:
;     mc_mktellspec
;
; PURPOSE:
;     Constructs the telluric correction spectra for SpeX.
;
; CATEGORY:
;     Spectroscopy
;
; CALLING SEQUENCE:
;     mc_mktellspec,std_wave,std_flux,std_error,std_mag,std_bminv,kernel,$
;                   scales,wvega,fvega,cfvega,cf2vega,vshift,tellcor,$
;                   tellcor_error,scvega,CANCEL=cancel
;
; INPUTS:
;     std_wave   - The wavelength array of the A0 V standard (in microns)
;     std_flux   - The flux array of the A0 V standard
;     std_error  - The error array of the A0 V standard
;     std_mag    - The magnitude of the A0 V standard
;     std_bminv  - The (B-V) color of the A0 V standard
;     kernel     - The convolution kernel
;     scales     - An array of scale factors for the Vega model at 
;                  the wavelengths "wave"
;     wvega      - The wavelength array of the Vega model shifted to 
;                  radial velocity of the A0 V standard
;     fvega      - The flux array of the Vega model
;     fcvega     - The continuum flux array of the Vega model
;     fc2vega    - The fitted continuum flux array of the Vega model
;     vshift     - The radial velocity shift of the A0 V relative the Vega
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     CANCEL  - Set on return if there is a problem
;
; OUTPUTS:
;     tellcor       - The telluric correction spectrum
;     tellcor_error - The telluric correction error spectrum 
;     scvega        - The convolved and scaled Vega model sampling 
;                     at wave
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
;     Complicated but based entirely on W. Vacca's telluric program.  
;
; EXAMPLE:
;
; MODIFICATION HISTORY:
;     2002-02-06 - Written by M. Cushing, Institute for Astronomy, UH
;     2018-05-11 - Modified to allow for variable R spectra
;-
pro mc_mktellspec,std_wave,std_flux,std_error,std_mag,std_bminv,kernel,$
                  wvega,fvega,cfvega,cf2vega,vshift,tellcor,tellcor_error, $
                  scvega,VARRINFO=varrinfo,SCALES=scales,CANCEL=cancel

  cancel = 0

;  Check parameters
  
;  if n_params() lt 10 then begin
;     
;     print, 'Syntax - mc_tellspec,std_wave,std_flux,std_error,std_mag,$'
;     print, '                     std_bminv,kernel,wvega,fcvega,fc2vega,$'
;     print, '                     vshift,tellcor,tellcor_error,scvega,$'
;     print, '                     SCALES=scales,CANCEL=cancel'
;     cancel = 1
;     return
;     
;  endif
;  cancel = mc_cpar('mc_mktellspec',std_wave,1,'Std_wave',[1,2,3,4,5],1)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',std_flux,2,'Std_flux',[1,2,3,4,5],1)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',std_error,3,'Std_errpr',[1,2,3,4,5],1)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',std_mag,4,'Std_mag',[1,2,3,4,5], 0)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',std_bminv,5,'Std_bminv',[1,2,3,4,5],0)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',kernel,6,'Kernel',[1,2,3,4,5],1)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',wvega,7,'Wvega',[1,2,3,4,5],1)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',cfvega,8,'Cfvega',[1,2,3,4,5],1)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',cf2vega,9,'Cf2vega',[1,2,3,4,5],1)
;  if cancel then return
;  cancel = mc_cpar('mc_mktellspec',vshift,10,'Vshift',[1,2,3,4,5],0)
;  if cancel then return
  
;  Shift the Vega model to the A0 V 
  
  rshift = 1.0 + (vshift/2.99792458D5)
  swvega = wvega*rshift
  
     
;  Now do the convolution.  Check to see whether it is the fixed R
;  case or not first.

  if keyword_set(VARRINFO) then begin

;  Interpolate the scale array onto the Vega sampling if requested.
     
     if keyword_set(SCALES) then begin
        
        trim = mc_nantrim(std_wave,2)
        linterp,std_wave[trim],scales[trim],swvega,rscales,MISSING=0
        
     endif else rscales=make_array(n_elements(swvega),VALUE=1.0)
     
     kwvega = (varrinfo.(0))[*,0]
     nshort  = (varrinfo.(0))[*,1]

     nwaves = n_elements(kwvega)
             
     nfvconv = dblarr(nwaves,/NOZERO)
     cfvconv = dblarr(nwaves,/NOZERO)
     cfvconv = dblarr(nwaves,/NOZERO)

     match,wvega,kwvega,aidx
     for i = 0,nwaves-1 do begin

        nlong = n_elements(kernel)
        ratio = nlong/float(nshort[i])
        x = (findgen(nshort[i]/2)+1)*ratio
        
        xx = [nlong/2-reverse(x),nlong/2,nlong/2+x]
        
        linterp,findgen(nlong),kernel,xx,newkernel
        newkernel = newkernel/total(newkernel)
        
;        tnfvconv = convol( (fvega/cfvega-1.0) * rscales, newkernel ) + 1.0
;        tcfvconv = convol( (cfvega/cf2vega-1.0) * rscales, newkernel ) + 1.0
;        tcfvconv = temporary(tcfvconv)*cf2vega

;        nfvconv[i] = tnfvconv[aidx[i]]
;        cfvconv[i] = tcfvconv[aidx[i]]
        
        
        lidx = (aidx[i]-nshort[i]/2)
        ridx = (aidx[i]+nshort[i]/2)

        int = (fvega[lidx:ridx]/cfvega[lidx:ridx]-1.0)*rscales[lidx:ridx]
        nfvconv[i] = total(int*reverse(newkernel))+1.0

        int = (cfvega[lidx:ridx]/cf2vega[lidx:ridx]-1.0)*rscales[lidx:ridx]
        cfvconv[i] = (total(int*reverse(newkernel))+1.0)*cf2vega[aidx[i]]

        
     endfor

     fvconv  = nfvconv*cfvconv
        
;  Now interpolate wvconv and fvconv onto the sampling of the A0 V

     linterp,kwvega*rshift,fvconv,std_wave,rfvconv        
             
  endif else begin

;  Determine the range over which to convolve the Vega model
  
     wmin = min(std_wave,/NAN,MAX=wmax)
     zv   = where(swvega ge wmin and swvega le wmax,nwave)
  
;  Now buffer that range based on the size of the kernel
  
     nkernel = n_elements(kernel)
     idx     = lindgen(nwave+2*nkernel)+zv[0]-nkernel
     
;  Interpolate the scale array onto the Vega sampling if requested.
     
     if keyword_set(SCALES) then begin
        
        trim = mc_nantrim(std_wave,2)
        linterp,std_wave[trim],scales[trim],wvega[idx],rscales
        
     endif else rscales=1.
          
     nfvconv = convol( (fvega[idx]/cfvega[idx]-1.0) * rscales, kernel ) + 1.0
     cfvconv = convol( (cfvega[idx]/cf2vega[idx]-1.0) * rscales, kernel ) + 1.0
     cfvconv = temporary(cfvconv)*cf2vega[idx]

     fvconv  = nfvconv*cfvconv

;  Now interpolate wvconv and fvconv onto the sampling of the A0 V

     linterp,swvega[idx],fvconv,std_wave,rfvconv
     
  endelse

;  Determine scale factors for flux calibration and reddening
  
  vegamag  = 0.03
  vegabmv  = 0.00
  magscale = 10.0^(-0.4*(std_mag-vegamag))
  ebmv     = (std_bminv - vegabmv) > 0.0 ; to prevent reddening the spectrum
  avred    = 3.10*ebmv

;  Redden the convolved Vega model
  
  mc_redden,std_wave,rfvconv,ebmv,fvred,CANCEL=cancel
  if cancel then return
  
;  Scale to the observed V mag of the A0V star
  
  fvred = temporary(fvred)*( magscale*(10.0^(0.4*avred)) )
  
  scvega = fvred

;  Calculate the ratio: model/observations
    
  z  = where(finite(std_flux) eq 1,count)
  wa = std_wave[z]
  fa = std_flux[z]
  ea = std_error[z]
  fv = fvred[z]
  
  z  = where(fa gt 0.0D)
  wa = wa[z]
  fa = fa[z]
  ea = ea[z]
  fv = fv[z]
  
  t  = fv/fa
  te = sqrt( (fv/fa^2)^2 * ea^2 )

  mc_interpspec,wa,t,std_wave,tellcor,tellcor_error,IYERROR=te
  
end


