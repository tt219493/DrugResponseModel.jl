"""Functions to plot Supplementary figure 5, 6, and 7, HCC1143, 21MT1, and MDAMB157 data overlayed with model predictions and accumulated cell death predictions."""


function plot_fig1_data(concs, g1data, tite, G, subPlabel, palet, time)
    p = Plots.plot(
        time,
        g1data,
        lw = 5,
        legend = :topleft,
        label = ["control" "$(concs[2]) nM" "$(concs[3]) nM" "$(concs[4]) nM" "$(concs[5]) nM" "$(concs[6]) nM" "$(concs[7]) nM" "$(concs[8]) nM"],
        fg_legend = :transparent,
        palette = palet,
        title = tite,
        titlefont = Plots.font("Helvetica", 14),
        legendfont = Plots.font("Helvetica", 11),
        guidefont = Plots.font("Helvetica", 14),
        xtickfont = Plots.font("Helvetica", 14),
        ytickfont = Plots.font("Helvetica", 14),
        xlabel = "Time [hr]",
        xticks = 0:24.0:96.0,
        ylabel = "Normalized Cell Counts",
        bottom_margin = 1.25cm,
        top_margin = 1.25cm,
        left_margin = 1.25cm,
        right_margin = 1.25cm,
    )
    annotate!(-0.5, 1.5, Plots.text(subPlabel, :black, :left, Plots.font("Helvetica Bold", 15)))
    ylims!((0.0, 4))
    p
end

function plot_fig1_percG1(concs, g1data, tite, G, subPlabel, palet, time)
    p = Plots.plot(
        time,
        g1data,
        lw = 5,
        legend = :topleft,
        label = ["control" "$(concs[4]) nM" "$(concs[5]) nM" "$(concs[6]) nM" "$(concs[7]) nM" "$(concs[8]) nM"],
        fg_legend = :transparent,
        palette = palet,
        title = tite,
        titlefont = Plots.font("Helvetica", 14),
        legendfont = Plots.font("Helvetica", 11),
        guidefont = Plots.font("Helvetica", 14),
        xtickfont = Plots.font("Helvetica", 14),
        ytickfont = Plots.font("Helvetica", 14),
        xlabel = "Time [hr]",
        xticks = 0:24.0:96.0,
        ylabel = "G1 Percentage",
        bottom_margin = 1.25cm,
        top_margin = 1.25cm,
        left_margin = 1.25cm,
        right_margin = 1.25cm,
    )
    annotate!(-0.5, 1.5, Plots.text(subPlabel, :black, :left, Plots.font("Helvetica Bold", 15)))
    ylims!((0.0, 1.0))
    p
end

""" Plot data G1 percentage and total cell counts, without the fits."""
function figure500()
    ENV["GKSwstype"]="nul"
    cellLine = "MDA-MB-157"
    tensor, names, concs, conds = DrugResponseModel.mda_all()
    t = LinRange(0.0, 96, size(tensor)[2])

    gmshort = zeros(size(tensor)[2], 6, 6, 2) # datapoints x concs x drugs x g1/g2
    for i=1:2
        gmshort[:, 1, :, i] .= tensor[i, :, 1, :]
        gmshort[:, 2:end, :, i] .= tensor[i, 4:end, :, :]
    end

    p1 = plot_fig1_data(concs[1], gmshort[:, :, 1, 1] .+ gmshort[:, :, 1, 2], string(cellLine, " treated with ", names[1]), "", "A", :PuBu_8, t)
    p2 = plot_fig1_data(concs[2], gmshort[:, :, 2, 1] .+ gmshort[:, :, 2, 2], string(cellLine, " treated with ", names[2]), "", "", :PuBu_8, t)
    p3 = plot_fig1_data(concs[3], gmshort[:, :, 3, 1] .+ gmshort[:, :, 3, 2], string(cellLine, " treated with ", names[3]), "", "", :PuBu_8, t)
    p4 = plot_fig1_data(concs[4], gmshort[:, :, 4, 1] .+ gmshort[:, :, 4, 2], string(cellLine, " treated with ", names[4]), "", "", :PuBu_8, t)
    p5 = plot_fig1_data(concs[5], gmshort[:, :, 5, 1] .+ gmshort[:, :, 5, 2], string(cellLine, " treated with ", names[5]), "", "", :PuBu_8, t)
    p6 = plot_fig1_data(concs[6], gmshort[:, :, 6, 1] .+ gmshort[:, :, 6, 2], string(cellLine, " treated with ", names[6]), "", "", :PuBu_8, t)

    p7 = plot_fig1_percG1(concs[1], gmshort[:, :, 1, 1] ./ (gmshort[:, :, 1, 1] .+ gmshort[:, :, 1, 2]), string(cellLine, " treated with ", names[1]), "", "B", :PuBu_8, t)
    p8 = plot_fig1_percG1(concs[2], gmshort[:, :, 2, 1] ./ (gmshort[:, :, 2, 1] .+ gmshort[:, :, 2, 2]), string(cellLine, " treated with ", names[2]), "", "", :PuBu_8, t)
    p9 = plot_fig1_percG1(concs[3], gmshort[:, :, 3, 1] ./ (gmshort[:, :, 3, 1] .+ gmshort[:, :, 3, 2]), string(cellLine, " treated with ", names[3]), "", "", :PuBu_8, t)
    p10 = plot_fig1_percG1(concs[4], gmshort[:, :, 4, 1] ./ (gmshort[:, :, 4, 1] .+ gmshort[:, :, 4, 2]), string(cellLine, " treated with ", names[4]), "", "", :PuBu_8, t)
    p11 = plot_fig1_percG1(concs[5], gmshort[:, :, 5, 1] ./ (gmshort[:, :, 5, 1] .+ gmshort[:, :, 5, 2]), string(cellLine, " treated with ", names[5]), "", "", :PuBu_8, t)
    p12 = plot_fig1_percG1(concs[6], gmshort[:, :, 5, 1] ./ (gmshort[:, :, 6, 1] .+ gmshort[:, :, 6, 2]), string(cellLine, " treated with ", names[6]), "", "", :PuBu_8, t)

    figure1 = Plots.plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, size = (2600, 800), layout = (2, 6))
    Plots.savefig(figure1, string("SupplementaryFigure567_data_", cellLine, ".svg"))
end


##############################  FUNCTIONS TO PLOT THE FITS  ################################
""" To plot the fits and accumulated cell death for each cell line, we do the following:
1. tensor, names, concs, conds = DrugResponseModel.__cellLineName__()
where __cellLineName__ could be one of [hcc_all, mt1_all, mda_all]
2. imporing the estimated parameters according to the cell line, one of [ps_hcc, ps_mt1, ps_mda] below.
3. DrugResponseModel.figure70(tensor, names, concs, conds, ps)"""

# 6-drug fits
ps_hcc = [34.9432, 27.829, 0.291144, 0.0244977, 2.64026, 0.115686, 0.442475, 3.61059e-5, 0.422046, 0.330667, 9.47844e-6, 3.39363e-6, 0.000705232, 2.34289e-5, 1.23165e-5, 2.20203e-6, 2.92093, 8.09313e-6, 159.166, 13.0877, 1.5532, 0.00387504, 3.63611, 0.260283, 0.450801, 0.0250362, 0.367939, 0.252831, 0.00107609, 0.000115707, 0.00260766, 6.82032e-5, 4.10667e-5, 9.84008e-6, 1.46092, 0.000134889, 0.741738, 2.4431, 3.9629, 0.000761489, 3.94527, 0.348368, 0.331174, 2.27071, 0.0737219, 0.591972, 0.00185578, 0.0162289, 0.00154473, 2.32892e-5, 1.1137e-7, 5.03267e-5, 7.37255e-7, 0.00531612, 1.46012, 4.8298, 0.517965, 0.08571, 0.736769, 0.0322651, 0.290752, 3.31035, 0.0641698, 0.292762, 9.08443e-5, 5.09103e-5, 0.000280376, 0.00737825, 0.000304891, 0.00129649, 3.0795e-5, 4.0636e-5, 86.4606, 0.338272, 3.88389, 0.00102644, 3.58318, 0.000195763, 0.104922, 3.96811, 0.24616, 0.657727, 0.000600609, 0.0132196, 0.000113768, 3.56294e-6, 4.42152e-6, 4.73675e-5, 8.75234e-6, 2.35412e-5, 3.716, 1.19482, 0.564743, 0.0361228, 3.98035, 0.110105, 0.403263, 3.96816, 0.421521, 0.640074, 0.0878646, 0.000286413, 0.000384056, 1.16993e-5, 1.07933e-5, 0.00215579, 8.21145e-5, 0.000489024, 3.98591, 0.588244, 3.98874, 0.268013, 0.441755, 3.99003, 0.365446, 0.505571]
ps_mt1 = [1.55185, 1.45433, 3.99842, 0.00724615, 2.46356, 0.102305, 2.60303, 3.3045, 1.04112, 2.71372, 1.82752e-5, 0.0057077, 1.04167e-5, 0.445341, 5.64574e-5, 3.24302e-5, 8.55308e-7, 3.00144e-5, 19.1045, 0.698229, 1.04623, 0.086005, 1.00304, 1.84451e-6, 2.55208, 3.29383, 0.606597, 2.67352, 3.3935e-6, 5.37281e-7, 0.000554417, 0.00117723, 1.21694e-5, 7.34184e-5, 1.15653e-5, 4.54135e-6, 220.841, 14.6972, 3.83869, 0.0975135, 2.15286, 0.302349, 2.5924, 3.29021, 0.438998, 2.65367, 0.000445606, 7.23863e-6, 0.00012159, 1.74178e-5, 1.34838e-5, 5.1107e-5, 3.84569e-6, 3.03796e-5, 12.5217, 2.23514, 0.161317, 0.249599, 2.47102, 0.0467338, 0.0740153, 1.24226, 0.495188, 0.67336, 3.46356e-7, 3.85531e-6, 4.22593e-5, 9.41142e-8, 6.0454e-7, 2.56087e-7, 4.66992e-7, 2.07443e-7, 51.6852, 2.10108, 0.0332774, 0.00824266, 0.00151619, 0.000457234, 2.60334, 3.28454, 0.0951307, 2.10065e-5, 0.0132314, 0.000176011, 0.646514, 3.2941, 9.48263e-5, 1.80983e-6, 1.39552e-6, 2.5178e-7, 1.51555, 2.32919, 1.65229, 0.0336268, 0.329872, 0.000369634, 2.53846, 4.65599e-5, 0.420307, 4.96256e-5, 4.27684e-6, 2.02799e-6, 1.51012e-7, 5.99119e-7, 2.30595e-5, 3.27434e-7, 3.62375e-6, 9.15975e-8, 3.99449, 0.603004, 2.4585, 3.99946, 2.5756, 3.28379, 0.512648, 2.64776]
ps_mda = [2.01648, 2.7393, 0.381298, 0.282195, 0.0426087, 0.00438904, 3.99969, 0.38872, 3.99991, 2.56698e-7, 2.74086e-7, 3.53968e-7, 4.66013e-7, 0.31455, 8.04555e-7, 1.79967e-7, 1.42038e-8, 0.0394145, 3594.01, 38.8183, 3.97541, 3.98819, 0.0427238, 1.01873, 0.0825609, 0.365346, 3.99997, 0.237293, 9.11982e-8, 2.1046e-7, 1.09047e-8, 1.41151e-8, 6.54729e-9, 2.89062e-8, 1.86293e-6, 1.22863e-8, 658.205, 2.11387, 3.99996, 0.150384, 0.0246126, 4.0, 3.7199, 0.296986, 3.99558, 0.619582, 9.13771e-8, 4.40758e-8, 2.34502e-8, 0.804942, 3.73842e-7, 1.71125e-8, 8.63841e-7, 3.34926e-8, 10.9042, 4.84963, 3.99996, 0.0861652, 0.0428161, 0.24123, 0.0676982, 0.127572, 3.1579, 0.367814, 4.89605e-7, 0.000408088, 1.35047e-8, 8.0359e-7, 1.64854e-7, 7.10794e-9, 1.41167e-7, 2.52954e-8, 247.586, 42.2574, 0.699003, 0.0740976, 0.00144902, 3.9916, 0.0833487, 0.338742, 3.97948, 0.175912, 7.82846e-7, 7.6491e-8, 0.0300662, 9.54405e-8, 2.03674e-6, 9.53742e-8, 1.03249e-6, 3.34538e-8, 0.642888, 7.62552, 3.99996, 0.117966, 0.344893, 0.236656, 0.080841, 0.306802, 3.9999, 0.807695, 3.52664e-7, 1.25166e-8, 7.37586e-8, 4.40688e-9, 1.28515e-8, 0.0340858, 1.90609e-7, 8.55822e-9, 3.99998, 3.99998, 0.123506, 3.99037, 3.99998, 0.270454, 4.0, 0.780807]

""" Helper for plotting time series fits. """
function plot_fig1(concs, g1, g1data, tite, G, subPlabel, palet, time)
    p = Plots.plot(
        time,
        g1,
        lw = 4,
        legend = :topleft,
        label = ["control" "$(concs[4]) nM" "$(concs[5]) nM" "$(concs[6]) nM" "$(concs[7]) nM" "$(concs[8]) nM"],
        fg_legend = :transparent,
        palette = palet,
        title = tite,
        titlefont = Plots.font("Helvetica", 14),
        legendfont = Plots.font("Helvetica", 11),
        guidefont = Plots.font("Helvetica", 14),
        xtickfont = Plots.font("Helvetica", 14),
        ytickfont = Plots.font("Helvetica", 14),
        xlabel = "time [hr]",
        xticks = 0:24.0:96.0,
        ylabel = "$G cell number",
        bottom_margin = 1.25cm,
        top_margin = 1.25cm,
        left_margin = 1.25cm,
        right_margin = 1.25cm,
    )
    Plots.plot!(time, g1data, lw = 4, linestyle = :dot, label = ["" "" "" "" "" "" ""])
    annotate!(-0.5, 1.5, Plots.text(subPlabel, :black, :left, Plots.font("Helvetica Bold", 15)))
    ylims!((0.0, 3.0))
    p
end

""" Helper function to plot accumulated dead cells. """
function plot_dead_acc(concs, drugs, efcs, siz)

    g = zeros(siz, 8, 6, 8) # total
    t = LinRange(0.0, 96.0, siz)
    d = zeros(siz, 6, 6) # datapoints x concs x drugs
    ls = [1, 4, 5, 6, 7, 8]
    for i = 1:6
        k = 1
        for j in ls
            g[:, j, i, 1], g[:, j, i, 2], g[:, j, i, 3], g[:, j, i, 4], g[:, j, i, 5], g[:, j, i, 6], g[:, j, i, 7], g[:, j, i, 8] =
                DrugResponseModel.predictD(efcs[:, j, i], efcs[:, 1, i], t)
            d[:, k, i] =
                efcs[9, j, i] .* g[:, j, i, 1] .+ efcs[10, j, i] .* g[:, j, i, 2] .+ efcs[11, j, i] .* g[:, j, i, 3] .+
                efcs[12, j, i] .* g[:, j, i, 4] .+ efcs[13, j, i] .* g[:, j, i, 5] .+ efcs[14, j, i] .* g[:, j, i, 6] .+
                efcs[15, j, i] .* g[:, j, i, 7] .+ efcs[16, j, i] .* g[:, j, i, 8]
            k += 1
        end
    end
    intg = zeros(siz, 6, 6)
    for i = 1:6
        for j = 1:6
            intg[:, j, i] = cumul_integrate(t, d[:, j, i])
        end
    end
    function single_dead(intt, i)
        p1 = Plots.plot(
            t,
            intt[:, :, i],
            labels =  ["control" "$(concs[i][4]) nM" "$(concs[i][5]) nM" "$(concs[i][6]) nM" "$(concs[i][7]) nM" "$(concs[i][8]) nM"],
            title = drugs[i],
            lw = 2,
            ylabel = " ",
            xlabel = "Time [hr]",
            palette = :YlOrRd_6,
            legend = :left,
            titlefont = Plots.font("Helvetica", 14),
            legendfont = Plots.font("Helvetica", 11),
            guidefont = Plots.font("Helvetica", 14),
            xtickfont = Plots.font("Helvetica", 14),
            ytickfont = Plots.font("Helvetica", 14),
            bottom_margin = 1.5cm,
            fg_legend = :transparent,
            top_margin = 1.5cm,
            left_margin = 1.25cm,
            right_margin = 1.25cm,
        )
        ylims!((-0.05, 2.0))
        return p1
    end

    p = [single_dead(intg, j) for j=1:6]
    return p

end

""" To plot the fits and accumulated cell death for each cell line, we do the following:
1. tensor, names, concs, conds = DrugResponseModel.__cellLineName__()
where __cellLineName__ could be one of [hcc_all, mt1_all, mda_all]
2. imporing the estimated parameters according to the cell line, one of [ps_hcc, ps_mt1, ps_mda] above.
3. DrugResponseModel.figure70(tensor, names, concs, conds, ps)"""

function figure501(tensor, names, concs, conds, ps)
    ENV["GKSwstype"]="nul"
    cellLine = "MDA-MB-157"
    cs = zeros(8, length(concs))
    for i=1:length(concs)
        cs[:, i] = concs[i]
    end
    efcs = getODEparams(ps, cs)
    t = LinRange(0.0, 96, size(tensor)[2])
    Gshort = zeros(size(tensor)[2], 6, 6, 2)
    cc = [1, 4, 5, 6, 7, 8]

    for (i, drug) in enumerate(names)
        for j=1:6
            Gshort[:, j, i, 1], Gshort[:, j, i, 2], _ = DrugResponseModel.predict(efcs[:, cc[j], i], efcs[:, 1, i], t)
        end
    end

    # params at EC50
    ec50 = zeros(16, 6)
    conc_ec50 = zeros((1, 6))
    k=1
    for i=1:6
        conc_ec50[1, i] = ps[k]
        k+=18
    end
    ec50 = getODEparams(ps, conc_ec50)[:, 1, :]

    # phase durations
    # @ control
    gi = zeros(2, 2, 6) # g1/g2 x control/ec50 x drugs
    gi[1, 1, :] .= (2 ./ efcs[1, 1, :] .+ 2 ./ efcs[2, 1, :] .+ 2 ./ efcs[3, 1, :] .+ 2 ./ efcs[4, 1, :])
    gi[2, 1, :] .= (5 ./ efcs[5, 1, :] .+ 5 ./ efcs[6, 1, :] .+ 5 ./ efcs[7, 1, :] .+ 5 ./ efcs[8, 1, :])

    # @ ec50
    gi[1, 2, :] .= (2 ./ ec50[1, :] .+ 2 ./ ec50[2, :] .+ 2 ./ ec50[3, :] .+ 2 ./ ec50[4, :])
    gi[2, 2, :] .= (5 ./ ec50[5, :] .+ 5 ./ ec50[6, :] .+ 5 ./ ec50[7, :] .+ 5 ./ ec50[8, :])

    gmshort = zeros(size(tensor)[2], 6, 6, 2) # datapoints x concs x drugs x g1/g2
    for i=1:2
        gmshort[:, 1, :, i] .= tensor[i, :, 1, :]
        gmshort[:, 2:6, :, i] .= tensor[i, :, 4:8, :]
    end

    # cell deaths
    deathContG1 = zeros(4, 6)
    deathEC50G1 = zeros(4, 6)
    deathContG1[1, :] .= (efcs[9, 1, :]) ./ (efcs[9, 1, :] .+ efcs[1, 1, :])
    deathEC50G1[1, :] .= (ec50[9, :]) ./ (ec50[9, :] .+ ec50[1, :])
    for i = 2:4
        deathContG1[i, :] .= (1 .- deathContG1[i - 1, :]) .* (efcs[i + 8, 1, :]) ./ (efcs[i + 8, 1, :] .+ efcs[i, 1, :])
        deathEC50G1[i, :] .= (1 .- deathEC50G1[i - 1, :]) .* (ec50[i + 8, :]) ./ (ec50[i + 8, :] .+ ec50[i, :])
    end
    deathContG2 = zeros(4, 6)
    deathEC50G2 = zeros(4, 6)
    deathContG2[1, :] .= (efcs[13, 1, :]) ./ (efcs[13, 1, :] .+ efcs[5, 1, :])
    deathEC50G2[1, :] .= (ec50[13, :]) ./ (ec50[13, :] .+ ec50[5, :])
    for i = 14:16
        deathContG2[i - 12, :] = (1 .- deathContG2[i - 13, :]) .* (efcs[i, 1, :]) ./ (efcs[i, 1, :] .+ efcs[i - 8, 1, :])
        deathEC50G2[i - 12, :] = (1 .- deathEC50G2[i - 13, :]) .* (ec50[i, :]) ./ (ec50[i, :] .+ ec50[i - 8, :])
    end

    p0 = Plots.plot(legend = false, grid = false, foreground_color_subplot = :white, top_margin = 1.5cm)
    p1 = plot_fig1(concs[1], Gshort[:, :, 1, 1], gmshort[:, :, 1, 1], string(" ", names[1]), "G1", "", :PuBu_6, t)
    p2 = plot_fig1(concs[1], Gshort[:, :, 1, 2], gmshort[:, :, 1, 2], string(" ", names[1]), "S-G2", "", :PuBu_6, t)
    p3 = plot_fig1(concs[2], Gshort[:, :, 2, 1], gmshort[:, :, 2, 1], string(" ", names[2]), "G1", "", :PuBu_6, t)
    p4 = plot_fig1(concs[2], Gshort[:, :, 2, 2], gmshort[:, :, 2, 2], string(" ", names[2]), "S-G2", "", :PuBu_6, t)
    p5 = plot_fig1(concs[3], Gshort[:, :, 3, 1], gmshort[:, :, 3, 1], string(" ", names[3]), "G1", "", :PuBu_6, t)
    p6 = plot_fig1(concs[3], Gshort[:, :, 3, 2], gmshort[:, :, 3, 2], string(" ", names[3]), "S-G2", "", :PuBu_6, t)
    p7 = plot_fig1(concs[4], Gshort[:, :, 4, 1], gmshort[:, :, 4, 1], string(" ", names[4]), "G1", "", :PuBu_6, t)
    p8 = plot_fig1(concs[4], Gshort[:, :, 4, 2], gmshort[:, :, 4, 2], string(" ", names[4]), "S-G2", "", :PuBu_6, t)
    p9 = plot_fig1(concs[5], Gshort[:, :, 5, 1], gmshort[:, :, 5, 1], string(" ", names[5]), "G1", "", :PuBu_6, t)
    p10 = plot_fig1(concs[5], Gshort[:, :, 5, 2], gmshort[:, :, 5, 2], string(" ", names[5]), "S-G2", "", :PuBu_6, t)
    p11 = plot_fig1(concs[6], Gshort[:, :, 6, 1], gmshort[:, :, 6, 1], string(" ", names[6]), "G1", "", :PuBu_6, t)
    p12 = plot_fig1(concs[6], Gshort[:, :, 6, 2], gmshort[:, :, 6, 2], string(" ", names[6]), "S-G2", "", :PuBu_6, t)
    p = plot_dead_acc(concs, names, efcs, size(Gshort)[1])

    figure1 = Plots.plot(p1, p3, p5, p7, p9, p11, p2, p4, p6, p8, p10, p12, p[1], p[2], p[3], p[4], p[5], p[6], size = (3100, 1300), layout = (3, 6))
    Plots.savefig(figure1, string("SupplementaryFigure567_fits_", cellLine, ".svg"))
end
