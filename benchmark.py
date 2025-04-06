import gc
import time
import statistics
from logging import DEBUG, WARNING, INFO

from lbvs_lib.classes import NMOD_POLY_TYPE, Commitment, OPENING_TYPE, ctypes
from lbvs_lib.compile import shared_library, MODP, DEGREE, WIDTH
from lbvs_lib.return_code_table import ReturnCodeTable
from lbvs_lib.shuffle import Shuffle
from lbvs_lib.utils import new_random_question, get_all_voting_combinations, random
from lbvs_lib.primitives import flint_rand, commitment_scheme
from lbvs_lib.logger import set_level, VERBOSE, LOGGER

def print_results(results, labels, n_voters):
    means = {label: statistics.mean(results[label]) for label in labels}
    medians = {label: statistics.median(results[label]) for label in labels}
    stdevs = {label: statistics.stdev(results[label]) for label in labels}

    for label in labels:
        LOGGER.warning(f"Time for {label} for {n_voters} voters: {means[label]}")
        LOGGER.warning(f"Median time for {label} for {n_voters} voters: {medians[label]}")
        LOGGER.warning(f"Standard deviation for {label} for {n_voters} voters: {stdevs[label]}")

def test_shuffle(**kwargs):
    voters_tests = kwargs["voters_tests"]
    n_executions = kwargs["n_executions"]

    for n_msgs in voters_tests:
        m = []
        for i in range(n_msgs):
            message = NMOD_POLY_TYPE()
            shared_library.nmod_poly_init(message, MODP)
            shared_library.commit_sample_short(message)
            m.append(message)
        differences = 0
        for i in range(n_executions):
            LOGGER.log(VERBOSE, f"EXECUTION {i}")
            before = time.time()
            m_ = Shuffle.shuffle(shared_library, m, n_msgs)
            after = time.time()
            for i in range(n_msgs):
                shared_library.nmod_poly_clear(m_[i])
            differences += (after - before)
        differences /= n_executions
        LOGGER.warning(f"Time for {n_msgs} messages: {differences}")
        for i in range(n_msgs):
            shared_library.nmod_poly_clear(m[i])

def cal_n_voting_options(answers, min, max):
    import math
    n = 0
    for i in range(min, max + 1):
        n += math.factorial(answers) / (math.factorial(i) * math.factorial(answers - i))
    return int(n)

def test_prf(**kwargs):
    answer_tests = kwargs["answers_tests"]
    n_executions = kwargs["n_executions"]

    blinding_key = NMOD_POLY_TYPE()
    shared_library.nmod_poly_init(blinding_key, MODP)
    shared_library.nmod_poly_randtest(blinding_key, flint_rand, DEGREE)


    key = ReturnCodeTable.new_key()

    for answer_size in answer_tests:
        question = new_random_question(len_a=answer_size, len_min=1, len_max=answer_size)
        differences = 0
        voting_opt = cal_n_voting_options(answer_size, 1, answer_size)
        for i in range(n_executions):
            before = time.time()
            table = ReturnCodeTable.compute_table(key, blinding_key, question)
            after = time.time()
            assert len(table) == voting_opt
            differences += (after - before)
        differences /= n_executions
        LOGGER.warning(f"Time for {answer_size} answers with {voting_opt} distinct options: {differences}")

def test_algorithms(**kwargs):
    from lbvs_lib.scheme_algorithms_bench import benchmark as algorithms_benchmark

    voters_tests = kwargs["voters_tests"]
    n_executions = kwargs["n_executions"]

    algs = ("SETUP", "REGISTER", "CAST", "CODE", "COUNT", "VERIFY")

    for n_voters in voters_tests:
        values = {
            alg: [] for alg in algs
        }
        for i in range(n_executions):
            LOGGER.log(VERBOSE, f"EXECUTION {i}")
            results = algorithms_benchmark(n_voters)
            for j, alg in enumerate(algs):
                values[alg].append(results[j])
        print_results(values, algs, n_voters)

def test_helios(**kwargs):
    label = "HELIOS"
    from plainscript.helios_benchmark import helios_benchmark
    test_prot(label, helios_benchmark, **kwargs)

def test_lbvs(**kwargs):
    label = "LBVS"
    from lbvs_lib.protocol_bench import protocol_benchmark
    test_prot(label, protocol_benchmark, **kwargs)

def test_prot(label, prot_function, answers_tests, voters_tests, n_executions, n_questions=1):
    phase_name = ("SETUP", "REGISTER", "VOTING", "COUNTING")
    algorithms = ("SETUP", "REGISTER", "CAST", "CODE", "COUNT", "VERIFY")

    for answer_size in answers_tests:
        questions = [new_random_question(len_a=answer_size, len_min=1, len_max=answer_size) for _ in range(n_questions)]
        comb_per_question = [list(get_all_voting_combinations(question['answers'], question['min'], question["max"]))
                             for question in questions]

        for n_voters in voters_tests:
            LOGGER.info(f"ANSWER SIZE: {answer_size} --- VOTERS: {n_voters}")

            results = {phase: [] for phase in phase_name}
            results_alg = {alg: [] for alg in algorithms}

            for j in range(n_executions):
                LOGGER.info(f"EXECUTION {j}")
                votes = [[list(random.choice(comp)) for comp in comb_per_question] for _ in range(n_voters)]
                results_iter, results_iter_alg = prot_function(n_voters, questions, votes)

                for i, phase in enumerate(phase_name):
                    results[phase].append(results_iter[i])
                for i, alg in enumerate(algorithms):
                    results_alg[alg].append(results_iter_alg[i])
                gc.collect()

            print_results(results, phase_name, n_voters)
            print_results(results_alg, algorithms, n_voters)


def test_register_with_rct(**kwargs):
    from lbvs_lib.protocol_bench import benchmark_registration_with_rct

    voters_tests = kwargs["voters_tests"]
    n_executions = kwargs["n_executions"]

    for n_voters in voters_tests:
        differences = [0, 0]
        for i in range(n_executions):
            LOGGER.log(VERBOSE, f"EXECUTION {i}")
            _, (result_reg, result_rct) = benchmark_registration_with_rct(n_voters,
                                                                          [new_random_question(len_a=8, len_max=8, len_min=1)
                                                                           for _ in range(1)])
            differences[0] += result_reg
            differences[1] += result_rct

        differences[0] /= n_executions
        differences[1] /= n_executions
        LOGGER.warning(f"Time for REGISTER for {n_voters} voters: {differences[0]}")
        LOGGER.warning(f"Time for Return Code Table generation for {n_voters} voters: {differences[1]}")


def test_shuffle_proof(**kwargs):
    voters_tests = kwargs["voters_tests"]
    n_executions = kwargs["n_executions"]

    for n_messages in voters_tests:
        LOGGER.info(f"Shuffle proof for {n_messages} messages")
        for i in range(n_executions):
            m = (NMOD_POLY_TYPE * n_messages)()
            com = (ctypes.POINTER(Commitment) * n_messages)()
            r = (OPENING_TYPE * n_messages)()

            ck = commitment_scheme.keygen()

            for i in range(n_messages):
                shared_library.nmod_poly_init(m[i], MODP)
                com[i] = shared_library.commit_ptr_init()
                shared_library.commit_sample_short(m[i])
                _, r[i] = commitment_scheme.commit(ck, m[i], only_r=True, commit_ref=com[i])

            _m = Shuffle.shuffle(shared_library, m, n_messages)
            LOGGER.info("Prover is running")
            proof = Shuffle.prover(shared_library, commitment_scheme.scheme, com, m, _m, r, ck,
                                   commitment_scheme.rand, n_messages)
            LOGGER.info("Verifier is running")
            result = Shuffle.verifier(shared_library, proof[0], proof[1], proof[2], proof[3], proof[4],
                                      commitment_scheme.scheme, proof[5], proof[6], com, _m, proof[7], ck, n_messages)

            if not result:
                LOGGER.error("Shuffle failed")
            else:
                LOGGER.info("Shuffle successful")
            commitment_scheme.keyfree(ck)

            for i in range(n_messages):
                shared_library.commit_free(com[i])
                shared_library.commit_ptr_free(com[i])
                shared_library.nmod_poly_clear(m[i])
                shared_library.nmod_poly_clear(_m[i])
                for j in range(WIDTH):
                    for k in range(2):
                        shared_library.nmod_poly_clear(r[i][j][k])
            Shuffle.proof_clear(shared_library, *proof, n_messages)

if __name__ == "__main__":
    import os
    import django

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
    django.setup()

    # set logging level
    # 10 = DEBUG
    set_level(VERBOSE)

    params = {
        "answers_tests": [8],
        "voters_tests": [2**i for i in range(4, 9, 2)],
        "n_executions": 3,
    }

    tests = {
        "shuffle": True,
        "prf": True,
        "helios": True,
        "lbvs": True,
        "algorithms": True,
        "shuffle_proof": False,
        "register_with_rct": True,
    }

    for test_name, value in tests.items():
        if value:
            exec(f"test_{test_name}(**{params})")