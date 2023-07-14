class AlignerCDR3:
    def __init__(self, gap_positions = (3, 4, -4, -3)):
        self.gap_positions = gap_positions

    @staticmethod
    def __pad(s1, s2, i, d) -> tuple[str]:
        if (d < 0):
            return (s1[:i] + ('-' * d) + s1[i:], s2)
        else:
            return (s1, s2[:i] + ('-' * d) + s2[i:])
        
    def pad(self, s1, s2) -> tuple[tuple[str]]:
        d = len(s1) - len(s2)
        if (d == 0):
            return tuple(tuple(s1, s2))
        else:
            return tuple(AlignerCDR3.__pad(s1, s2, p, d) for p in self.gap_positions)
        
    def __score(self, s1, s2) -> float:
        if s1 != s2:
            return 0.0
        else:
            return 1.0
        
    def score(self, s1, s2) -> float:
        return sum(self.__score(*x) for x in zip(s1, s2))
        