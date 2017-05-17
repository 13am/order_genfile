from argparse import ArgumentParser
import sys
import gzip
import numpy


MAX_SNPS_IN_BUFFER = 6


def parse_options():

    parser = ArgumentParser()

    parser.add_argument('--gen-in',
                        dest='genfile_in',
                        default=None,
                        help='Name of the input .gen file. Leave out if'
                        ' you want to read the input from stdin.')

    parser.add_argument('--gen-out',
                        dest='genfile_out',
                        default=None,
                        help='Name of the output .gen file. Leave out if'
                        ' you want to write the output to stdout.')

    parser.add_argument('--sample-in',
                        dest='sample_in',
                        default=None,
                        help='Name of the input .sample file.')

    parser.add_argument('--sample-out',
                        dest='sample_out',
                        default=None,
                        help='Name of the output .sample file.')

    parser.add_argument('--order',
                        dest='order',
                        default=None,
                        help='Specify the order of the output data.'
                        ' ID_1 ID_2 per line.'
                        ' First line -> first 3 probabilities etc.')

    parser.add_argument('--model-sample',
                        dest='model_sample',
                        default=None,
                        help='Match the order of this sample file'
                        ' when outputting data.')

    parser.add_argument('--allow-subsetting',
                        action='store_true',
                        dest='allow_subsetting',
                        default=False,
                        help='Exclude any samples found in the input sample '
                             'file but not in the file order or model sample '
                             'files.')

    options = parser.parse_args()
    return options


def print_finish_and_stop():
    sys.stderr.write('\nProgram finished with success.\n')
    sys.exit()


def print_error_and_stop(msg):
    main_msg = '\nError: {}\nTerminating the program because of the error.\n'
    sys.stderr.write(main_msg.format(msg))
    sys.exit()


def read_gen_lines(lines, n_samples):

    # make the numpy array, verify all lines have the same number of columns
    array = numpy.array(([i.strip().split(' ') for i in lines]))
    n = len(array[0])
    for i in array:
        if len(i) != n:
            print_error_and_stop('The number of genotypes in the input'
                                 ' is not constant.')

    # the line may or may not have the chromosome code
    i_genotypes = 5
    if len(array[0]) % 3 == 0:
        i_genotypes = 6
    non_genotype_cols = array[..., :i_genotypes]
    genotype_cols = array[..., i_genotypes:]

    # always make sure the number of probabilities is 3 times N(of samples)
    if len(genotype_cols[0]) != 3 * n_samples:
        msg = ('Line for the SNP \n"{}"\ndid not have 3*{}'
               ' genotype probabilities')
        print_error_and_stop(msg.format(' '.join(non_genotype_cols),
                                        n_samples))

    # use numpy to reshape the genotype columns
    # now they are in [variant][probabilities]
    # change into     [variant][sample][probabilities]
    n_snps = len(lines)
    genotypes = genotype_cols.reshape(n_snps, n_samples, 3)

    return non_genotype_cols, genotypes


def read_sample_order(fname, is_sample_file=None):
    input_sample_order = []
    with open(fname, 'r') as ip:
        for line in ip.readlines():
            ln = line.strip().split()
            sample_id = ' '.join(ln[:2])
            input_sample_order += [sample_id]

    if is_sample_file:
        input_sample_order = input_sample_order[2:]

    # fail if duplicates are found
    if len({i: i for i in input_sample_order}) != len(input_sample_order):
        msg = 'The file "{}" contains duplicates.'
        print_error_and_stop(msg.format(fname))

    return tuple(input_sample_order)


def _write_lines(op, lines, n_samples_input, index_array):
    snp_info_lines, genotypes = read_gen_lines(lines, n_samples_input)
    new_indexing = numpy.ix_(range(0, len(genotypes)),
                             index_array)
    ordered_genotypes = genotypes[new_indexing]
    ordered_genotypes = ordered_genotypes.reshape(len(lines),
                                                  len(index_array) * 3)
    op_data = numpy.column_stack((snp_info_lines,
                                  ordered_genotypes))
    op_lines = [' '.join(i) for i in op_data]
    op.write('\n'.join([i for i in op_lines]) + '\n')
    return len(op_lines)


def write_output(ip, op, n_samples_input, index_array):
    counter = 0
    buffered_lines = []
    for line in ip:
        buffered_lines.append(line)
        if len(buffered_lines) == MAX_SNPS_IN_BUFFER:
            counter += _write_lines(op, buffered_lines,
                                    n_samples_input, index_array)
            buffered_lines = []
    if len(buffered_lines) > 0:
        counter += _write_lines(op, buffered_lines,
                                n_samples_input, index_array)

    return counter


def main():
    options = parse_options()

    # read the input sample order
    input_sample_order = read_sample_order(options.sample_in,
                                           is_sample_file=True)

    # read the output sample order
    new_sample_order_fname = options.model_sample
    is_sample_file = True
    if options.model_sample is None:
        new_sample_order_fname = options.new_order
        is_sample_file = False
    new_sample_order = read_sample_order(new_sample_order_fname,
                                         is_sample_file=is_sample_file)
    new_sample_dict = {v: i for i, v in enumerate(new_sample_order)}

    # make a dict from old -> new indexes for the genotypes
    old_to_new_index = {}
    for i_old, sample_id in enumerate(input_sample_order):
        try:
            i_new = new_sample_dict[sample_id]
            old_to_new_index[i_old] = i_new
        except KeyError:
            if options.allow_subsetting:
                pass
            else:
                msg = 'The sample "{}" was not in listed in "{}"' \
                      ' and the --allow-subsetting flag was not' \
                      ' in use.'
                msg = msg.format(sample_id,
                                 new_sample_order_fname)
                print_error_and_stop(msg)

    # write the new sample file
    if options.sample_out is None:
        print_error_and_stop('Please specify --sample-out.')
    with open(options.sample_out, 'w') as op:
        with open(options.sample_in, 'r') as ip:
            old_lines = ip.readlines()
            new_lines = [None for i in new_sample_order]
            for i_old, l in enumerate(old_lines[2:]):
                try:
                    i_new = old_to_new_index[i_old]
                    new_lines[i_new] = l
                except KeyError:
                    continue
            new_lines = old_lines[:2] + new_lines
            op.write('\n'.join([i.strip() for i in new_lines]) + '\n')

    # convert the dict to an index array
    new_to_old_index = {i_new: i_old for i_old, i_new in
                        old_to_new_index.items()}
    index_array = [None for i in new_to_old_index]
    for new_index, old_index in new_to_old_index.items():
        index_array[new_index] = old_index
    if None in index_array:
        print_error_and_stop('Error in making the index array')
    index_array = tuple(index_array)

    # figure out where and how to read the input genfile
    if options.genfile_in is None:
        ip = sys.stdin
    elif options.genfile_in.endswith('.gz'):
        ip = gzip.open(options.genfile_in, 'rb')
    else:
        ip = open(options.genfile_in, 'r')

    # set output destination
    if options.genfile_out is None:
        op = sys.stdout
    else:
        op = open(options.genfile_out, 'w')

    n_samples_input = len(input_sample_order)
    n_written = write_output(ip, op, n_samples_input, index_array)
    op.flush()

    opname = 'STDOUT'
    if op is not sys.stdout:
        op.close()
        opname = options.genfile_out
    msg = '\nWrote {} samples and {} genotype lines to "{}"" and "{}"\n'
    msg = msg.format(len(index_array), n_written,
                     options.sample_out, opname)
    sys.stderr.write(msg)
    print_finish_and_stop()


if __name__ == '__main__':
    main()
