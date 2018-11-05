#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <ctime>
using namespace std;

struct node{
    float s;
    int id;
};
bool cmp(node a, node b){
    return a.s < b.s;
}

const int N = 11000;
const int M = 900;
float rep[N][M];
int version[N], K;
node s[N][N];

void compute(float rep[N][M], node s[N][N], int ref_cnt, int cnt){
    for(int i=cnt == ref_cnt ? 0 : ref_cnt; i<cnt; i++)
        for(int j=0; j<ref_cnt; j++){
            s[i][j].id = j;
            for(int k=0; k<K; k++){
                float v = rep[i][k] - rep[j][k];
                v *= v;
                s[i][j].s += v;
            }
        }
}

void calc_result(node s[N][N], int version[N], int ref_cnt, int cnt){
    float MAP = 0, MPat10 = 0, MR1 = 0;
    int que_cnt = ref_cnt == cnt ? cnt : (cnt - ref_cnt);
    for(int i=cnt == ref_cnt ? 0 : ref_cnt; i<cnt; i++){
        float AP = 0, Pat10 = 0, R1 = 0;
        int ver_cnt = 0;
        s[i][i].s = -1;
        sort(&s[i][0], &s[i][ref_cnt], cmp);
        for(int idx=(ref_cnt == cnt ? 1: 0), j=1; idx<ref_cnt; idx++, j++){
            int u = i, v = s[i][idx].id;
            // printf("%d %d ", version[u], version[v]);
            if(version[u] == version[v]){
                if(j <= 10)
                    Pat10 += 1;
                if(!ver_cnt)
                    R1 = j;
                ver_cnt += 1;
                AP += (float)ver_cnt / j;
            }
        }
        if(ver_cnt == 0)
            printf("%d ", i);
        MAP += AP / ver_cnt;
        MR1 += R1;
        MPat10 += Pat10;
    }
    
    printf("MAP=%.5f MPat10=%.5f MR1=%.5f\n", MAP / que_cnt, MPat10 / que_cnt / 10, MR1 / que_cnt);
}

int main(int argc, char* argv[])
{
    int ref_cnt = stoi(argv[2]), cnt = stoi(argv[3]);
    K = stoi(argv[4]);  
 
    memset(s, 0, sizeof(s));
    char vname[30];
    sprintf(vname, "%s/version.txt", argv[1]);
    char fname[30];
    sprintf(fname, "%s/data%d.txt", argv[1], K == 900 ? 1 : 0);
   
    FILE *fp = fopen(vname, "r");
    for(int i=0; i<N; i++)
        fscanf(fp, "%d", &version[i]);
    fclose(fp);
    
    fp = fopen(fname, "r");
    for(int i=0; i<N; i++)
        for(int j=0; j<K; j++)
            fscanf(fp, "%f", &rep[i][j]);
    fclose(fp);

    cout << "input finish\n";
    // start time
    int start_s=clock();
    compute(rep, s, ref_cnt, cnt);
    int stop_s=clock();
    // end time
    cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000/(cnt == ref_cnt ? cnt : (cnt - ref_cnt)) << "ms" << endl;
    calc_result(s, version, ref_cnt, cnt);
 
    return 0;
}
