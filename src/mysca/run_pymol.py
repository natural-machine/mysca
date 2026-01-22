"""
Example usage:

sca-pymol -s Soil14.scaffold_576820813_c1_40 \
    --pdb_dir out/structure/K00370 \
    --groups_dir out/sca/K00370/sca_groups \
    --outdir out/sca/K00370/images \
    --groups 0 1 2
"""

import argparse
import os
import sys
import shutil
import pymol
from pymol import cmd
import numpy as np
import imageio
from PIL import Image
import tqdm as tqdm
import json

from mysca.constants import SECTOR_COLORS

MO_COLOR = None
SF4_COLOR = None

DEFAULT_STRUCT_COLOR = "gray70"
DEFAULT_STRUCT_STYLE = "sticks"
DEFAULT_STRUCT_ALPHA = 0.5
DEFAULT_SECTOR_COLORS = SECTOR_COLORS
DEFAULT_SECTOR_STYLE = "spheres"

DEFAULT_BG_COLOR = "white"

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scaffold", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--modes", type=str, required=True)

    # parser.add_argument("--groups_dir", type=str, required=True)
    parser.add_argument("--groups", type=int, nargs="*", default=-1,
                        help="Group indices, (starting at 0) that correspond " \
                        "to subdirectories group_<idx> of groups_dir. If -1, " \
                        "produce plots for all groups.")
    parser.add_argument("--multisector", action="store_true", 
                        help="Plot sectors simultaneously on the same protein.")
    parser.add_argument("-r", "--reference", type=str, default=None,
                        help="If specified, align the input scaffold to this " \
                        "reference.")
    parser.add_argument("--features", type=str, default=None,
                        help="File with json objects to include as annotations.")
    # parser.add_argument("--scores_dir", type=str, default=None)
    parser.add_argument("--views", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--show_molybdenum", action="store_true")
    parser.add_argument("-o", "--outdir", type=str, default=None)
    parser.add_argument("-v", "--verbosity", type=int, default=1)

    return parser.parse_args(args)


def main(args):
    scaffold = args.scaffold
    pdb_dir = args.pdb_dir
    modes_fpath = args.modes
    group_idxs = args.groups
    multisector = args.multisector
    ref_scaffold = args.reference
    show_molybdenum = args.show_molybdenum
    outdir = args.outdir
    verbosity = args.verbosity
    views = args.views
    animate = args.animate
    nframes = args.nframes
    duration = args.duration
    features_fpath = args.features

    if ref_scaffold is None or ref_scaffold.lower() == "none":
        ref_scaffold = None

    if features_fpath:
        with open(features_fpath) as f:
            features = json.load(f)
    else:
        features = {}

    # Process the file detailing mode positions and scores. Set found_modes
    # to the list of integer values identifying each mode. 
    mode_data = np.load(modes_fpath)
    key_list = list(mode_data.keys())
    found_modes = np.sort(np.unique([int(k.split("_")[1]) for k in key_list]))    
    
    # Get all sectors/groups if group_idxs is specified as -1 at the cmd line
    # Check modes file and include all modes present
    if group_idxs[0] == -1:
        group_idxs = found_modes
    else:
        assert np.all([gi in found_modes for gi in group_idxs])

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    # Colors and styles
    struct_color = DEFAULT_STRUCT_COLOR
    struct_style = DEFAULT_STRUCT_STYLE
    struct_alpha = DEFAULT_STRUCT_ALPHA
    sector_colors = [_hex2color(x) for x in DEFAULT_SECTOR_COLORS]
    sector_style = DEFAULT_SECTOR_STYLE
    background_color = DEFAULT_BG_COLOR

    # Load the specified structure
    if verbosity:
        print("Scaffold:", scaffold)
    pdbfile = f"{pdb_dir}/{scaffold}.pdb"
    cmd.load(pdbfile, "struct")

    # Load a reference, if given
    if ref_scaffold:
        reffile = f"{pdb_dir}/{ref_scaffold}.pdb"
        cmd.load(reffile, "ref_struct")

    # Hide the loaded structure (and reference)
    cmd.hide("everything", "struct")
    if ref_scaffold:
        cmd.hide("everything", "ref_struct")
    
    # Set background color
    cmd.bg_color(background_color)

    # Show structure
    cmd.show(struct_style, "struct")
    cmd.color(struct_color, "struct")
    cmd.set(
        {"sticks": "stick_transparency"}.get(struct_style, DEFAULT_STRUCT_STYLE), 
        1 - struct_alpha, 
        "struct"
    )
    cmd.zoom(complete=1)

    if multisector:
        if verbosity:
            print(f"Plotting {scaffold} with all sectors...")
        
        plot_scaffold_with_multiple_sectors(
            scaffold, group_idxs, mode_data, 
            struct_color=struct_color, 
            sector_colors=sector_colors, 
            sector_style=sector_style, 
            ref_scaffold=ref_scaffold, 
            features=features,
            views=views,
            outdir=outdir, 
            verbosity=verbosity,
        )
    else:
        if verbosity:
            print(f"Plotting {scaffold} by sector...")
        
        plot_scaffold_by_sectors(
            scaffold, group_idxs, mode_data,
            struct_color=struct_color, 
            sector_colors=sector_colors, 
            sector_style=sector_style, 
            ref_scaffold=ref_scaffold, 
            features=features,
            views=views,
            animate=animate,
            nframes=nframes,
            duration=duration,
            outdir=outdir, 
            verbosity=verbosity,
        )
    
    if verbosity:
        print("Done!")


def plot_scaffold_by_sectors(
        scaffold, group_idxs, mode_data, *,
        struct_color, 
        sector_colors, 
        sector_style, 
        ref_scaffold, 
        outdir, 
        verbosity=1,
        features=None,
        views=True,
        animate=False, 
        **kwargs
):
    nframes = kwargs.get("nframes", 24)
    duration = kwargs.get("duration", 2.4)
    
    if nframes is None:
        nframes = 24
    if duration is None:
        duration = 2.4
    
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    for gidx in group_idxs:
        groupkey = f"sector_{gidx}_pdbpos_{scaffold}"
        sector_color = sector_colors[gidx]
        if groupkey in mode_data:
            group_selection = "group_selection"
            group = mode_data[groupkey]
            res_idxs = 1 + group
            selection_string = "resi " + "+".join(map(str, res_idxs))
            cmd.select(group_selection, selection_string)
            cmd.show(sector_style, group_selection)
            cmd.color(sector_color, group_selection)
        else:
            group_selection = None
            group = None
            if verbosity:
                print(f"Structure {scaffold} group {gidx} data not found: {groupkey}")
        
        # Load scores, if a directory is specified
        scorekey = f"sector_{gidx}_scores_{scaffold}"
        scores = None
        alphas = None
        if scorekey in mode_data:
            scores = mode_data[scorekey]
            MIN_ALPHA = 0.5
            svals = np.square(scores)
            s0, s1 = svals.min(), svals.max()
            a0, a1 = MIN_ALPHA, 1
            alphas = (a1 - a0) / (s1 - s0) * (svals - s1) + a1
        if alphas is not None:
            # Apply transparency per residue
            if verbosity > 1:
                print(f"Applying alphas [{alphas.min():.4g}, {alphas.max():.4g}]")
            for resi, alpha in zip(res_idxs, alphas):
                cmd.set(
                    {"spheres": "sphere_transparency"}.get(
                        sector_style, DEFAULT_SECTOR_STYLE
                    ), 
                    1 - alpha, 
                    f"{group_selection} and resi {resi}"
                )

        # Align the structure to the reference
        if ref_scaffold:
            cmd.align("struct", "ref_struct")
        cmd.center()
        cmd.zoom(complete=1)
        
        # Show extra features #TODO: generalize based on text file input
        if features:
            for item in features:
                _show_feature(
                    item["name"], item["struct"], item["selection_string"],
                    selection=item.get("selection", "everything"),
                    color=item.get("color", None),
                )
        
        # Save the primary plot
        cmd.png(f"{outdir}/{scaffold}_group{gidx}.png", dpi=300)

        # Save views of each side
        viewsdir = os.path.join(outdir, "views")
        if views:
            os.makedirs(viewsdir, exist_ok=True)
            for ri in range(4):
                cmd.png(f"{viewsdir}/{scaffold}_group{gidx}_view{ri}.png", dpi=300)
                cmd.rotate("y", 90, "struct")
                if ref_scaffold:
                    cmd.rotate("y", 90, "ref_struct")
        
        # Save animation
        if animate:
            RAY_FIRST = 1  # Better quality if 1. Faster if 0.
            seconds_per_frame = duration / nframes
            
            framesdir = f"{outdir}/frames/{scaffold}_group{gidx}_frames"
            os.makedirs(framesdir, exist_ok=True)

            for i in tqdm.trange(nframes, leave=False):
                cmd.turn("y", 360 / nframes)
                filename = os.path.join(framesdir, f"frame_{i:03d}.png")
                cmd.png(
                    filename, 
                    # width=800, height=800, 
                    dpi=300, ray=RAY_FIRST,
                )
            
            # Combine frames into a GIF
            frames = []
            for i in range(nframes):
                path = os.path.join(framesdir, f"frame_{i:03d}.png")
                im = Image.open(path).convert("RGBA")
                # Flatten the transparent background to white
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel("A"))
                frames.append(np.array(bg))
            
            # Save as GIF (no alpha channel, no ghosting)
            outfile = os.path.join(outdir, f"{scaffold}_group{gidx}.gif")
            imageio.mimsave(
                outfile,
                frames,
                duration=seconds_per_frame,  # seconds per frame
                loop=0,  # loop forever
                disposal=2,  # full-frame replace between frames
            )
            # shutil.rmtree(framesdir)
        
        # Reset
        if group_selection:
            cmd.hide(sector_style, group_selection)
            cmd.color(struct_color, group_selection)  # reset color
            cmd.delete(group_selection)

    return


def plot_scaffold_with_multiple_sectors(
        scaffold, group_idxs, mode_data, *, 
        struct_color, 
        sector_colors, 
        sector_style, 
        ref_scaffold, 
        outdir, 
        verbosity=1, 
        features=None,
        views=True,
):
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    sector_styles = [sector_style] * len(group_idxs)

    for i, gidx in enumerate(group_idxs):
        groupkey = f"sector_{gidx}_pdbpos_{scaffold}"
        sector_color = sector_colors[gidx]
        if groupkey in mode_data:
            group_selection = f"group_selection{i}"
            group = mode_data[groupkey]
            res_idxs = 1 + group
            selection_string = "resi " + "+".join(map(str, res_idxs))
            cmd.select(group_selection, selection_string)
            cmd.show(sector_styles[i], group_selection)
            cmd.color(sector_color, group_selection)
        else:
            group_selection = None
            group = None
            if verbosity:
                print(f"Group {gidx} data not found: {groupkey}")

        # Load scores, if a directory is specified
        scorekey = f"sector_{gidx}_scores_{scaffold}"
        scores = None
        alphas = None
        if scorekey in mode_data:
            scores = mode_data[scorekey]
            MIN_ALPHA = 0.5
            svals = np.square(scores)
            s0, s1 = svals.min(), svals.max()
            a0, a1 = MIN_ALPHA, 1
            alphas = (a1 - a0) / (s1 - s0) * (svals - s1) + a1
        if alphas is not None:
            # Apply transparency per residue
            if verbosity > 1:
                print(f"Applying alphas [{alphas.min():.4g}, {alphas.max():.4g}]")
            for resi, alpha in zip(res_idxs, alphas):
                cmd.set(
                    {"spheres": "sphere_transparency"}.get(
                        sector_style, DEFAULT_SECTOR_STYLE
                    ), 
                    1 - alpha, 
                    f"{group_selection} and resi {resi}"
                )
    
    # Align the structure to the reference
    if ref_scaffold:
        cmd.align("struct", "ref_struct")
    cmd.center()
    cmd.zoom(complete=1)

    # Show extra features
    if features:
        for item in features:
            _show_feature(
                item["name"], item["struct"], item["selection_string"],
                selection=item.get("selection", "everything"),
                color=item.get("color", None),
            )
        

    # Save the primary plot
        cmd.png(f"{outdir}/{scaffold}_groups_{",".join(
                [str(i) for i in group_idxs])}.png", dpi=300)
    
    # Save views of each side
    viewsdir = os.path.join(outdir, "views")
    if views:
        os.makedirs(viewsdir, exist_ok=True)
        for ri in range(4):
            cmd.png(f"{viewsdir}/{scaffold}_groups_{",".join(
                [str(i) for i in group_idxs])}_view{ri}.png", dpi=300)
            cmd.rotate("y", 90, "struct")
            if ref_scaffold:
                    cmd.rotate("y", 90, "ref_struct")
    
    return


def _hex2color(x):
    return "0x" + x[1:]


def _show_feature(
        name, struct, selection_string,
        selection="everything",
        color=None,
):
    cmd.select(name, f"{struct}/{selection_string}")
    cmd.show(selection, name)
    if isinstance(color, str):
        cmd.color(color, name)
    return

def _show_molybdenum(
        struct,
        color=None
):
    cmd.select("mo", f"{struct}/F/A/6MO`1302/MO")
    cmd.show("everything", "mo")
    if isinstance(color, str):
        cmd.color(color, "mo")
    return


def _show_sf4(
        struct,
        color=None,
):
    cmd.select("sf4", f"{struct}/G/A/SF4`1401/*")
    cmd.show("everything", "sf4")
    if isinstance(color, str):
        cmd.color(color, "sf4")
    return


def _show_cofactor(
        struct,
        color=None,
):
    cmd.select("cofactor", f"{struct}/D/A/MD1`1300/* {struct}/E/A/MD1`1301/*")
    cmd.show("sticks", "cofactor")
    if isinstance(color, str):
        cmd.color(color, "cofactor")
    return
    

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
