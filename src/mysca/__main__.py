import sys

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
