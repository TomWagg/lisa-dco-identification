import argparse
from cosmic.sample.stroopwafel import AdaptiveSampler, ParameterSpace, Parameter
from cosmic.sample.stroopwafel.presets import merging_dco
from cosmic.utils import parse_inifile

M1_MIN = { "NSWD": 4, "NSNS": 5, "BHWD": 14, "BHNS": 16, "BHBH": 19}
IS_INTERESTING = {
    "NSWD": merging_dco(kstar_1=[13], kstar_2=[10, 11, 12]),
    "NSNS": merging_dco(kstar_1=[13], kstar_2=[13]),
    "BHWD": merging_dco(kstar_1=[14], kstar_2=[10, 11, 12]),
    "BHNS": merging_dco(kstar_1=[14], kstar_2=[13]),
    "BHBH": merging_dco(kstar_1=[14], kstar_2=[14]),
}

def create_dco_population(metallicity, inifile_path, total_systems, batch_size, dco_type, nproc, output_path):

    def derive_params(sampled):
        """Provide binary parameters not drawn from the ParameterSpace."""
        return {'mass_2': sampled['mass_1'] * sampled['q'], "metallicity": metallicity}

    param_list = [
        Parameter('mass_1', M1_MIN[dco_type], 150.0, dist='kroupa'),
        Parameter('q', 0.0, 1.0, dist='uniform'),
        Parameter('porb', 10**(0.15), 10**(5.5),  dist='sana'),
        Parameter('ecc', 1e-9, 0.9999, dist='sana_ecc'),
    ]

    params = ParameterSpace(param_list)

    BSEDict, SSEDict, _, _, _, _ = parse_inifile(inifile_path)

    sampler = AdaptiveSampler(
        parameter_space=params,
        total_systems=total_systems,
        batch_size=batch_size,
        BSEDict=BSEDict,
        SSEDict=SSEDict,
        is_interesting=IS_INTERESTING[dco_type],
        derive_params=derive_params,
        reject_systems="default",
        nproc=nproc,
        n_generations=1,
        seed=117,
        only_save_hit_tables=True
    )
    result = sampler.run()

    print(f"Finished generating {total_systems} systems for metallicity {metallicity:.2e} and DCO type {dco_type}.")

    result.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create a population of DCOs using adaptive sampling.")
    parser.add_argument("--metallicity", type=float, required=True, help="Metallicity of the population.")
    parser.add_argument("--inifile", type=str, required=True, help="Path to the ini file with simulation parameters.")
    parser.add_argument("--total_systems", type=int, required=True, help="Total number of systems to generate.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of systems to generate per batch.")
    parser.add_argument("--dco_type", type=str, choices=["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"], required=True, help="Type of DCO to generate.")
    parser.add_argument("--nproc", type=int, required=True, help="Number of processes to use for parallelization.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated population data.")
    args = parser.parse_args()

    create_dco_population(
        metallicity=args.metallicity,
        inifile_path=args.inifile,
        total_systems=args.total_systems,
        batch_size=args.batch_size,
        dco_type=args.dco_type,
        nproc=args.nproc,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
