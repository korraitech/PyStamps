def ps_plot_tca(aps, aps_flag):
    """
    Translated from MATLAB function ps_plot_tca.m
    
    This function determines which APS correction array to use based on 
    the specified aps_flag (either a string or numeric), and returns:
      1) the correction array (aps_corr),
      2) a string describing the chosen correction (fig_name_tca),
      3) the numeric value of aps_flag (in case it was originally string).

    Parameters
    ----------
    aps : dict of numpy.ndarray
        Dictionary of arrays, each containing a particular APS correction 
        (e.g. aps['ph_tropo_linear'], aps['ph_tropo_powerlaw'], etc.).
        Keys match the naming in the legacy MATLAB code.
    aps_flag : int or str
        Either the numeric code or a string representing which correction 
        to retrieve.

    Returns
    -------
    aps_corr : numpy.ndarray
        The selected APS correction array.
    fig_name_tca : str
        A description of the correction used (for figure labeling, etc.).
    aps_flag : int
        The numeric representation of the aps_flag.
    """

    import numpy as np

    # -------------------------------------------------------------------------
    # If aps_flag is a string, convert it to the correct numeric code
    # -------------------------------------------------------------------------
    if isinstance(aps_flag, str):
        flag_map = {
            # linear
            'a_linear': 1, 'a_l': 1,
            # powerlaw
            'a_powerlaw': 2, 'a_p': 2,
            # meris
            'a_meris': 3, 'a_m': 3,
            # era-i / era5 (combined in the code)
            'a_erai': 4, 'a_e': 4,
            'a_erai-h': 5, 'a_eh': 5,
            'a_erai-w': 6, 'a_ew': 6,
            # wrf
            'a_wrf': 7, 'a_w': 7,
            'a_wrf-h': 8, 'a_wh': 8,
            'a_wrf-w': 9, 'a_ww': 9,
            # meris (non-interpolated)
            'a_meris-ni': 10, 'a_mi': 10,
            # powerlaw K map
            'a_powerlaw-k': 11, 'a_pk': 11,
            # modis
            'a_modis': 12, 'a_M': 12,
            'a_modis-ni': 13, 'a_MI': 13,
            # meris + ERA hydro
            'a_meris+a_erai-h': 14, 'a_m+a_eh': 14,
            'a_meris-ni+a_erai-h': 15, 'a_mi+a_eh': 15,
            # modis + ERA hydro
            'a_modis+a_erai-h': 16, 'a_M+a_eh': 16,
            'a_modis-ni+a_erai-h': 17, 'a_MI+a_eh': 17,
            # manual linear
            'a_linear-man': 18, 'a_lman': 18,
            # modis recal
            'a_recalmodis': 19, 'a_RM': 19,
            'a_recalmodis-ni': 20, 'a_RMI': 20,
            'a_recalmodis+a_erai-h': 21, 'a_RM+a_eh': 21,
            'a_recalmodis-ni+a_erai-h': 22, 'a_RMI+a_eh': 22,
            # meris + WRF hydro
            'a_meris+a_wrf-h': 23, 'a_m+a_wh': 23,
            'a_meris-ni+a_wrf-h': 24, 'a_mi+a_wh': 24,
            # modis + WRF hydro
            'a_modis+a_wrf-h': 25, 'a_M+a_wh': 25,
            'a_modis-ni+a_wrf-h': 26, 'a_MI+a_wh': 26,
            'a_recalmodis+a_wrf-h': 27, 'a_RM+a_wh': 27,
            'a_recalmodis-ni+a_wrf-h': 28, 'a_RMI+a_wh': 28,
            # merra
            'a_merra': 29,
            'a_merra2': 30,
            'a_merra-h': 31,
            'a_merra2-h': 32,
            'a_merra-w': 33,
            'a_merra2-w': 34,
            # gacos
            'a_gacos': 35,
            # narr
            'a_narr': 36,
            'a_narr-h': 37,
            'a_narr-w': 38,
            # era5 (reuse some era variable naming)
            'a_era5': 39,  # also 'a_e'
            'a_era5-h': 40,  # also 'a_eh'
            'a_era5-w': 41,  # also 'a_ew'
        }
        # Attempt to retrieve from our dictionary:
        if aps_flag in flag_map:
            aps_flag = flag_map[aps_flag]
        else:
            raise ValueError(f"aps_flag '{aps_flag}' not a valid APS option")

    # -------------------------------------------------------------------------
    # Determine which correction to return based on the numeric aps_flag
    # -------------------------------------------------------------------------
    if aps_flag == 1:
        # linear correction
        aps_corr = aps['ph_tropo_linear']
        fig_name_tca = ' (linear)'
    elif aps_flag == 2:
        # powerlaw correlation
        aps_corr = aps['ph_tropo_powerlaw']
        fig_name_tca = ' (powerlaw)'
    elif aps_flag == 3:
        # meris
        aps_corr = aps['ph_tropo_meris']
        fig_name_tca = ' (meris)'
    elif aps_flag == 4:
        # era / era5
        aps_corr = aps['ph_tropo_era']
        fig_name_tca = ' (era)'
    elif aps_flag == 5:
        # era hydro
        aps_corr = aps['ph_tropo_era_hydro']
        fig_name_tca = ' (era hydro)'
    elif aps_flag == 6:
        # era wet
        aps_corr = aps['ph_tropo_era_wet']
        fig_name_tca = ' (era wet)'
    elif aps_flag == 7:
        # wrf
        aps_corr = aps['ph_tropo_wrf']
        fig_name_tca = ' (wrf)'
    elif aps_flag == 8:
        # wrf hydro
        aps_corr = aps['ph_tropo_wrf_hydro']
        fig_name_tca = ' (wrf hydro)'
    elif aps_flag == 9:
        # wrf wet
        aps_corr = aps['ph_tropo_wrf_wet']
        fig_name_tca = ' (wrf wet)'
    elif aps_flag == 10:
        # meris no interp
        aps_corr = aps['ph_tropo_meris_no_interp']
        fig_name_tca = ' (meris)'
    elif aps_flag == 11:
        # powerlaw K
        aps_corr = aps['K_tropo_powerlaw']
        fig_name_tca = ' (powerlaw - spatial K map)'
    elif aps_flag == 12:
        # modis
        aps_corr = aps['ph_tropo_modis']
        fig_name_tca = ' (modis)'
    elif aps_flag == 13:
        # modis no interp
        aps_corr = aps['ph_tropo_modis_no_interp']
        fig_name_tca = ' (modis)'
    elif aps_flag == 14:
        # meris + ERA hydro
        ix_no_meris = (aps['ph_tropo_meris'].sum(axis=0) == 0)
        aps_corr = aps['ph_tropo_meris'] + aps['ph_tropo_era_hydro']
        # zero out columns where meris is empty
        aps_corr[:, ix_no_meris] = 0
        fig_name_tca = ' (meris + ERA hydro)'
    elif aps_flag == 15:
        # meris (no interp) + ERA hydro
        aps_corr = aps['ph_tropo_meris_no_interp'] + aps['ph_tropo_era_hydro']
        fig_name_tca = ' (meris + ERA hydro)'
    elif aps_flag == 16:
        # modis + ERA hydro
        ix_no_modis = (aps['ph_tropo_modis'].sum(axis=0) == 0)
        aps_corr = aps['ph_tropo_modis'] + aps['ph_tropo_era_hydro']
        aps_corr[:, ix_no_modis] = 0
        fig_name_tca = ' (modis + ERA hydro)'
    elif aps_flag == 17:
        # modis (no interp) + ERA hydro
        aps_corr = aps['ph_tropo_modis_no_interp'] + aps['ph_tropo_era_hydro']
        fig_name_tca = ' (modis + ERA hydro)'
    elif aps_flag == 18:
        # manual linear
        aps_corr = aps['strat_corr']
        fig_name_tca = ' (linear)'
    elif aps_flag == 19:
        # modis recal
        aps_corr = aps['ph_tropo_modis_recal']
        fig_name_tca = ' (modis recal)'
    elif aps_flag == 20:
        # modis recal no interp
        aps_corr = aps['ph_tropo_modis_no_interp_recal']
        fig_name_tca = ' (modis recal)'
    elif aps_flag == 21:
        # modis recal + ERA hydro
        ix_no_modis = (aps['ph_tropo_modis'].sum(axis=0) == 0)
        aps_corr = aps['ph_tropo_modis_recal'] + aps['ph_tropo_era_hydro']
        aps_corr[:, ix_no_modis] = 0
        fig_name_tca = ' (modis recal + ERA hydro)'
    elif aps_flag == 22:
        # modis recal no interp + ERA hydro
        aps_corr = aps['ph_tropo_modis_no_interp_recal'] + aps['ph_tropo_era_hydro']
        fig_name_tca = ' (modis recal + ERA hydro)'
    elif aps_flag == 23:
        # meris + WRF hydro
        ix_no_meris = (aps['ph_tropo_meris'].sum(axis=0) == 0)
        aps_corr = aps['ph_tropo_meris'] + aps['ph_tropo_wrf_hydro']
        aps_corr[:, ix_no_meris] = 0
        fig_name_tca = ' (meris + WRF hydro)'
    elif aps_flag == 24:
        # meris (no interp) + WRF hydro
        aps_corr = aps['ph_tropo_meris_no_interp'] + aps['ph_tropo_wrf_hydro']
        fig_name_tca = ' (meris + WRF hydro)'
    elif aps_flag == 25:
        # modis + WRF hydro
        ix_no_modis = (aps['ph_tropo_modis'].sum(axis=0) == 0)
        aps_corr = aps['ph_tropo_modis'] + aps['ph_tropo_wrf_hydro']
        aps_corr[:, ix_no_modis] = 0
        fig_name_tca = ' (modis + WRF hydro)'
    elif aps_flag == 26:
        # modis (no interp) + WRF hydro
        aps_corr = aps['ph_tropo_modis_no_interp'] + aps['ph_tropo_wrf_hydro']
        fig_name_tca = ' (modis + WRF hydro)'
    elif aps_flag == 27:
        # modis recal + WRF hydro
        ix_no_modis = (aps['ph_tropo_modis'].sum(axis=0) == 0)
        aps_corr = aps['ph_tropo_modis_recal'] + aps['ph_tropo_wrf_hydro']
        aps_corr[:, ix_no_modis] = 0
        fig_name_tca = ' (modis recal + WRF hydro)'
    elif aps_flag == 28:
        # modis recal (no interp) + WRF hydro
        aps_corr = aps['ph_tropo_modis_no_interp_recal'] + aps['ph_tropo_wrf_hydro']
        fig_name_tca = ' (modis recal + WRF hydro)'
    elif aps_flag == 29:
        # MERRA
        aps_corr = aps['ph_tropo_merra']
        fig_name_tca = ' (MERRA)'
    elif aps_flag == 30:
        # MERRA-2
        aps_corr = aps['ph_tropo_merra2']
        fig_name_tca = ' (MERRA-2)'
    elif aps_flag == 31:
        # MERRA hydro
        aps_corr = aps['ph_tropo_merra_hydro']
        fig_name_tca = ' (MERRA hydro)'
    elif aps_flag == 32:
        # MERRA-2 hydro
        aps_corr = aps['ph_tropo_merra2_hydro']
        fig_name_tca = ' (MERRA-2 hydro)'
    elif aps_flag == 33:
        # MERRA wet
        aps_corr = aps['ph_tropo_merra_wet']
        fig_name_tca = ' (MERRA wet)'
    elif aps_flag == 34:
        # MERRA-2 wet
        aps_corr = aps['ph_tropo_merra2_wet']
        fig_name_tca = ' (MERRA-2 wet)'
    elif aps_flag == 35:
        # GACOS
        aps_corr = aps['ph_tropo_gacos']
        fig_name_tca = ' (GACOS)'
    elif aps_flag == 36:
        # NARR
        aps_corr = aps['ph_tropo_narr']
        fig_name_tca = ' (NARR)'
    elif aps_flag == 37:
        # NARR hydro
        aps_corr = aps['ph_tropo_narr_hydro']
        fig_name_tca = ' (NARR hydro)'
    elif aps_flag == 38:
        # NARR wet
        aps_corr = aps['ph_tropo_narr_wet']
        fig_name_tca = ' (NARR wet)'
    elif aps_flag == 39:
        # ERA5
        aps_corr = aps['ph_tropo_era5']
        fig_name_tca = ' (era5)'
    elif aps_flag == 40:
        # ERA5 hydro
        aps_corr = aps['ph_tropo_era5_hydro']
        fig_name_tca = ' (era hydro5)'
    elif aps_flag == 41:
        # ERA5 wet
        aps_corr = aps['ph_tropo_era5_wet']
        fig_name_tca = ' (era wet5)'
    else:
        raise ValueError("Not a valid numeric APS option")

    return aps_corr, fig_name_tca, aps_flag
