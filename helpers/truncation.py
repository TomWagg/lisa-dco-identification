import initial_dists
from copy import copy
from scipy.integrate import quad



def get_f_trunc(
        m1_low, m1_upp=150, m1_uni_low=0.08, m1_uni_upp=150, m2_low=0.08,
        f_bin=initial_dists.offner_multiplicity,
        mass_ratio_pdf_function=lambda q: 1
    ):
    """Calculate the fraction of the total mass in a COSMIC population relative to the Universal population.

    This accounts for the fact that we only sampled binaries, with large primary masses and a minimum secondary mass.
    The fraction is calculated by integrating the IMF over the relevant mass ranges,
    weighted by the binary fraction and the expected companion mass.

    :math:`f_{\rm trunc} = \frac{\displaystyle\int_{m_1^{\rm min}}^{m_1^{\rm max}} \zeta(m_1)\,f_{\rm bin}(m_1) m_1 \int_{m_2^{\rm min}/m_1}^{1} (1 + q) p(q) \dd{q} \dd{m_1}}{\displaystyle\int_{m_1^{\rm min, IMF}}^{m_1^{\rm max}} \int_{m_2^{\rm min}/m_1}^{1} \qty[ f_{\rm bin}(m_1) \cdot m_1 (1 + q) + (1 - f_{\rm bin}(m_1)) m_1 ]  \zeta(m_1) p(q) \dd{q} \dd{m_1}}`

    Parameters
    ----------
    m1_low : `float`
        Lower limit on the sampled primary mass
    m1_upp : `float`, optional
        Upper limit on the sampled primary mass, by default 150 Msun
    m1_uni_low : `float`, optional
        Lower limit on the universal primary mass, by default 0.08 Msun (hydrogen burning limit)
    m1_uni_upp : `float`, optional
        Upper limit on the universal primary mass, by default 150 Msun
    m2_low : `float`, optional
        Lower limit on the sampled secondary mass, by default 0.08 Msun (hydrogen burning limit)
    f_bin : `function` or `float`, optional
        Binary fraction as a function of primary mass, by default follows Offner et al. (2023)
    mass_ratio_pdf_function : `function`, optional
        Function to calculate the mass ratio PDF, by default a uniform mass ratio distribution

    Returns
    -------
    f_trunc : `float`
        Fraction of the total mass in a COSMIC population relative to the Universal population.
    """ 
    # if the binary fraction is a float, we can just make it a function that returns that value
    if isinstance(f_bin, float):
        f_bin_val = copy(f_bin)
        f_bin = lambda m: f_bin_val

    # first, for normalisation purposes, we can find the integral with no cosmic cuts
    def full_integral(mass):
        primary_mass = initial_dists.IMF(mass) * mass
        
        # find the expected companion mass given the mass ratio pdf function
        expected_secondary_mass = quad(lambda q: q * mass_ratio_pdf_function(q), 0, 1)[0] * primary_mass
        
        single_stars = (1 - f_bin(mass)) * primary_mass
        binary_stars = f_bin(mass) * (primary_mass + expected_secondary_mass)
        return single_stars + binary_stars
    full_mass = quad(full_integral, m1_uni_low, m1_uni_upp)[0]
    
    # now we do a similar integral but for the cosmic regime
    def cosmic_integral(mass, m2_low, f_bin):
        # define the primary mass in the same way
        primary_mass = initial_dists.IMF(mass) * mass
        
        # find the fraction that are below the m2 mass cut
        f_below_m2low = quad(mass_ratio_pdf_function, 0, m2_low / mass)[0]
        
        # expectation value of the secondary mass given the m2 cut and mass ratio pdf function
        expected_secondary_mass = quad(lambda q: q * mass_ratio_pdf_function(q), m2_low / mass, 1)[0] * primary_mass
        
        # return total mass of binary stars that have m2 above the cut
        return f_bin(mass) * (1 - f_below_m2low) * (primary_mass + expected_secondary_mass)
    cosmic_mass = quad(cosmic_integral, m1_low, m1_upp, args=(m2_low, f_bin))[0]

    return cosmic_mass / full_mass