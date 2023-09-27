from collections import namedtuple
from sys import platform


if platform == 'linux' or platform == 'linux2':
    zcat_cmd = 'zcat'
elif platform == 'darwin':
    zcat_cmd = 'gzcat'
elif platform == 'win32':
    raise 'Cannot run pipe on Windows'


def vidjil_pre_cmd(vidjil_path: str = '../vidjil/vidjil-algo',
                   vidjil_germline: str = '../vidjil/germline/homo-sapiens.g',
                   result_path: str = './result',
                   chunk_id: str = '{#}') -> str:
    return f'{vidjil_path} -c detect -g {vidjil_germline} -2 -U -o {result_path} -b c{chunk_id} -'


def vidjil_post_cmd(vidjil_path: str = '../vidjil/vidjil-algo',
                    vidjil_germline: str = '../vidjil/germline/homo-sapiens.g',
                    result_path: str = './result') -> str:
    return f'{vidjil_path} -c clones -g {vidjil_germline} -2 --all -U -o {result_path} -'


def fastq_cmd(input_path: str | list[str] = './input.fastq.gz',
              n_reads: int = -1) -> str:
    if input_path.endswith('gz'):
        cat_cmd = zcat_cmd
    else:
        cat_cmd = 'cat'
    if isinstance(input_path, list):
        input_path = " ".join(input_path)
    cmd = f'{cat_cmd} {input_path}'
    if n_reads > 0:
        cmd = f'{cmd} | head -n {n_reads}'
    return cmd


def parallel_cmd(input: str = fastq_cmd(),
                 vidjil: str = vidjil_pre_cmd(),
                 cores: int = 4,
                 log_path: str = './result/log.txt') -> str:
    if log_path:
        return f'{input} | parallel -j {cores} --pipe -L 4 --joblog {log_path} --round-robin "{vidjil}"'
    else:
        return f'{input} | parallel -j {cores} --pipe -L 4 --round-robin "{vidjil}"'


def pipeline_pre_cmd(input_path: str | list[str] = './sample.fastq.gz',
                     n_reads: int = -1,
                     vidjil_path: str = '../vidjil/vidjil-algo',
                     vidjil_germline: str = '../vidjil/germline/homo-sapiens.g',
                     result_path: str = './result',
                     cores: int = 4,
                     log_path: str = './result/vidjil_log.txt') -> str:
    return {'input': input_path,
            'output': [f'{result_path}/{c}.detected.vdj.fa' for c in range(1, cores + 1)],
            'cmd': parallel_cmd(input=fastq_cmd(input_path=input_path,
                                                n_reads=n_reads),
                                vidjil=vidjil_pre_cmd(vidjil_path=vidjil_path,
                                                      vidjil_germline=vidjil_germline,
                                                      result_path=result_path,
                                                      chunk_id='{#}'),
                                cores=cores,
                                log_path=log_path)}


def pipeline_post_cmd(vidjil_output: str | list[str] = './result/*.fa',
                      vidjil_path: str = '../vidjil/vidjil-algo',
                      vidjil_germline: str = '../vidjil/germline/homo-sapiens.g',
                      result_path: str = './result') -> str:
    if isinstance(vidjil_output, list):
        vidjil_output = ' '.join(vidjil_output)
    return f'cat {vidjil_output} | {vidjil_post_cmd(vidjil_path, vidjil_germline, result_path)}' 


VidjilRead = namedtuple('VidjilRead', 'id flags seq')


_EXAMPLE = '''>A019:398:HLJSX5:43:26 + VJ 	1 98 132 150	 seed IGL SEG_+ 1.00e-01 9.71e-02/2.95e-03
GATCTCCATCCTCTTGGTCACGCTCCCTAAGAGCCTTCGGCTTCTTTCTCCCAGTTCTGGTCTCTGGGGCTGGC
CGCCGGTGGGCGGGAACAGCATCGA
CTCTCCTTCCCAC'''


def parse_vidjil_read(lines : str | list[str]) -> VidjilRead:
    if not isinstance(lines, list):
        lines = lines.split('\n')
    header = lines[0].split(' ')
    read_id = header[0][1:]
    vidjil_flags = [_.strip() for _ in ' '.join(header[1:]).split('\t')]
    sequence = ''.join(lines[1:])
    return VidjilRead(read_id, vidjil_flags, sequence)


#print(process_vidjil_read(_EXAMPLE))
#print(process_vidjil_read(_EXAMPLE.split('\n')))

#def read_vidjil(files : str | list[str] = ['./result/c1.detected.vdj.fa']):
    


print(pipeline_cmd())
