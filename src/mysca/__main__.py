import sys

def run_prealign():
    from mysca.run_prealign import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def run_preprocessing():
    from mysca.run_preprocessing import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def run_sca():
    from mysca.run_sca import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def run_pymol():
    from mysca.run_pymol import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def run_plots():
    from mysca.run_plots import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)
