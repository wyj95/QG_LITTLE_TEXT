#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<windows.h>
#define BEGIN begin=clock()
#define END end=clock()

/**
 *  @name        : void InsertSort(int *a,int n);
 *  @description : ���������㷨
 *  @param       : ����ָ�� a, ���鳤�� n
 */
void InsertSort(int *a, long int n);

/**
 *  @name        : void MergeArray(int *a,int begin,int end);
 *  @description : �鲢���򣨺ϲ����飩
 *  @param       : ����ָ��a���������begin�������յ�end
 */
void MergeArray(int *a,long int begin,long int end);

/**
 *  @name        : void MergeSort(int *a,int begin,int end);
 *  @description : �鲢����
 *  @param       : ����ָ��a���������begin�������յ�end
 */
void MergeSort(int *a,long int begin,long int end);

/**
 *  @name        : void QuickSortRecursion(int *a, int begin, int end);
 *  @description : �������򣨵ݹ�棩
 *  @param       : ����ָ��a���������begin�������յ�end
 */
void QuickSortRecursion(int *a, long int begin, long int end);


/**
 *  @name        : void QuickSort(int *a,long int begin, long int end)
 *  @description : �������򣨷ǵݹ�棩
 *  @param       : ����ָ��a�����鳤��size
 */
void QuickSort(int *a,long int begin, long int end);

/**
 *  @name        : long int Partition(int *a, int begin, int end)
 *  @description : �������������ţ�
 *  @param       : ����ָ��a���������begin�������յ�end
 */
long int Partition(int *a, long int begin, long int end);


/**
 *  @name        : void CountSort(int *a, int size)
 *  @description : ��������
 *  @param       : ����ָ��a�����鳤��size
 */
void CountSort(int *a, long int size);
 
/**
 *  @name        : void RadixCountSort(int *a,int size)
 *  @description : ������������
 *  @param       : ����ָ��a�����鳤��size
 */
void RadixCountSort(int *a,long int size);

/**
 *  @name        : void swap(int *a,int *b)
 *  @description : ����ab��ֵ 
 *  @param       : ��������ָ��a��b 
 */
void swap(int *a, int *b);

/**
 *  @name        : void ColorSortMain()
 *  @description : ��ɫ���������� 
 *  @param       : ����ָ��a��ֻ��0��1��2Ԫ�أ������鳤��size
 */
void ColourSortMain();

/**
 *  @name        : void ColorSort(int *a,long int size)
 *  @description : ��ɫ����
 *  @param       : ����ָ��a��ֻ��0��1��2Ԫ�أ������鳤��size
 */
void ColourSort(int *a,long int size);

/**
 *  @name        : int FindKthMain()
 *  @description : ���������� 
 *  @param       : None
 */
void FindKthMain();

/**
 *  @name        : int FindKth(int *a, int key, int begin, int end)
 *  @description : ��һ�������������ҵ���key��/С����
 *  @param       : ����ָ��a��Ŀ�����key��key���ڵ�����(begin,end) Ҳ������������Χ 
 */
int FindKth(int *a, long int key, long int begin, long int end);


/**
 *  @name        : void random_number(int *p,long int size)
 *  @description : �������size�����ִ洢��p��
 *  @param       : ����ָ��p�������Сsize 
 */
void random_number(int *p,long int size);

/**
 *  @name        : void TestBigData(long int size)
 *  @description : �Ը��������㷨���г�������� 
 *  @param       : �����Сsize 
 */
void TextBigData(long int size);

/**
 *  @name        : void TestLittleData()
 *  @description : �Ը��������㷨���ж��(1e5)������(100)���� 
 *  @param       : None
 */
void TextLittleData();

/**
 *  @name        : void text_sort()
 *  @description : �������������һ�β�������ȷ��� 
 *  @param       : None
 */
void TextSort();

/**
 *  @name        : void ReadFile()
 *  @description : ���ļ��ж������ݲ������Ż�ȥ 
 *  @param       : None
 */
void ReadFile();

/**
 *  @name        : void WriteFile()
 *  @description : �����ɵ��������������ļ� 
 *  @param       : None
 */
void WriteFile();

/**
 *  @name        : void WriteRandFile()
 *  @description : д������� 
 *  @param       : None
 */
void WriteRandFile();

/**
 *  @name        : void GetLongInt(long int *a)
 *  @description : ��ȡһ��С��1000 000 ����0 ���� 
 *  @param       : a (����ȡ���ĵ�ַ) 
 */
void GetLongInt(long int *a);

/**
 *  @name        : void show()
 *  @description : ������ 
 *  @param       : None 
 */
void Show();
